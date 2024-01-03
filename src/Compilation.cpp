/****************************************************************************
 *
 *    Copyright (c) 2024 Vivante Corporation
 *
 *    Permission is hereby granted, free of charge, to any person obtaining a
 *    copy of this software and associated documentation files (the "Software"),
 *    to deal in the Software without restriction, including without limitation
 *    the rights to use, copy, modify, merge, publish, distribute, sublicense,
 *    and/or sell copies of the Software, and to permit persons to whom the
 *    Software is furnished to do so, subject to the following conditions:
 *
 *    The above copyright notice and this permission notice shall be included in
 *    all copies or substantial portions of the Software.
 *
 *    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 *    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 *    FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 *    DEALINGS IN THE SOFTWARE.
 *
 *****************************************************************************/

#include "Compilation.h"

#include <sys/fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <algorithm>
#include <filesystem>
#include <string>

#include "Types.h"
#include "Utils.h"

namespace vsi::android::sl {

Compilation::~Compilation() {
    close(cacheFd_);
}

int Compilation::setPreference(PreferenceCode preference) {
    if (finished_) {
        LOGE("Compilation::setPreference called after compilation finished");
        return ANEURALNETWORKS_BAD_STATE;
    }
    preference_ = preference;
    return ANEURALNETWORKS_NO_ERROR;
}

int Compilation::setPriority(PriorityCode priority) {
    if (finished_) {
        LOGE("Compilation::setPriority called after compilation finished");
        return ANEURALNETWORKS_BAD_STATE;
    }
    priority_ = priority;
    return ANEURALNETWORKS_NO_ERROR;
}

int Compilation::setTimeout(Duration duration) {
    if (finished_) {
        LOGE("Compilation::setTimeout called after compilation finished");
        return ANEURALNETWORKS_BAD_STATE;
    }
    timeoutDuration_ = duration;
    return ANEURALNETWORKS_NO_ERROR;
}

int Compilation::setCaching(const fs::path& cacheDir, const uint8_t* token) {
    if (finished_) {
        LOGE("Compilation::setCaching called after compilation finished");
        return ANEURALNETWORKS_BAD_STATE;
    }

    // The filename includes kByteSizeOfCacheToken * 2 characters.
    std::string filename(ANEURALNETWORKS_BYTE_SIZE_OF_CACHE_TOKEN * 2UL, '0');
    for (size_t i = 0; i < ANEURALNETWORKS_BYTE_SIZE_OF_CACHE_TOKEN; i++) {
        filename[i * 2] = 'A' + (token[i] & 0x0F);  // NOLINT(*-magic-numbers)
        filename[i * 2 + 1] = 'A' + (token[i] >> 4);
    }

    fs::path cacheFile = cacheDir / filename;
    int fd = open(cacheFile.c_str(), O_CREAT | O_EXCL | O_RDWR, 0);
    if (fd == -1) {
        if (errno == EEXIST) {
            // The file exists, delete it and try again.
            if (unlink(cacheFile.c_str()) == -1) {
                // No point in retrying if the unlink failed.
                LOGE("Compilation::setCaching error unlinking cache file %s: %s (%d)",
                     cacheFile.c_str(), strerror(errno), errno);
                return ANEURALNETWORKS_BAD_DATA;
            }
            // Retry now that we've unlinked the file.
            fd = open(cacheFile.c_str(), O_CREAT | O_EXCL | O_RDWR, 0);
        }
        if (fd == -1) {
            LOGE("Compilation::setCaching error creating cache file %s: %s (%d)", cacheFile.c_str(),
                 strerror(errno), errno);
            return ANEURALNETWORKS_BAD_DATA;
        }
    }

    return Compilation::setCaching(fd, token);
}

int Compilation::setCaching(int fd, const uint8_t* token) {
    if (finished_) {
        LOGE("Compilation::setCaching called after compilation finished");
        return ANEURALNETWORKS_BAD_STATE;
    }

    struct stat cacheStat;
    if (fstat(fd, &cacheStat) < 0) {
        LOGE("Compilation::setCaching failed to stat cache file: %s (%d)", strerror(errno), errno);
        return ANEURALNETWORKS_BAD_DATA;
    }

    if ((cacheStat.st_mode & (S_IRUSR | S_IWUSR)) == 0) {
        LOGE("Compilation::setCaching cache file not in RW mode");
        return ANEURALNETWORKS_BAD_DATA;
    }

    int cacheFd = dup(fd);
    if (cacheFd == -1) {
        LOGE("Compilation::setCaching failed to dup cache fd: %s (%d)", strerror(errno), errno);
        return ANEURALNETWORKS_BAD_DATA;
    }
    cacheFd_ = cacheFd;

    std::copy_n(token, ANEURALNETWORKS_BYTE_SIZE_OF_CACHE_TOKEN, cacheToken_.data());

    return ANEURALNETWORKS_NO_ERROR;
}

int Compilation::finish() {
    if (cacheFd_ != -1) {
        struct stat cacheStat;
        if (fstat(cacheFd_, &cacheStat) < 0) {
            LOGE("Compilation::finish failed to stat cache file: %s (%d)", strerror(errno), errno);
            return ANEURALNETWORKS_BAD_DATA;
        }

        off_t size = cacheStat.st_size;
        if (size > kNBGMagic.size()) {
            LOGD("Compilation::finish read cache file of %zd KB", size / 1024);
            cacheBuffer_.resize(size);

            ssize_t readSize = 0;
            size_t bufferOffset = 0;
            do {
                readSize = read(cacheFd_, cacheBuffer_.data() + bufferOffset, size);
                bufferOffset += readSize;
            } while (readSize > 0);

            if (readSize < 0) {
                LOGE("Compilation::finish failed to read cache file: %s (%d)", strerror(errno),
                     errno);
                return ANEURALNETWORKS_BAD_DATA;
            }

            // Check if cache file is NBG format.
            for (size_t i = 0; i < kNBGMagic.size(); i++) {
                char symbol = static_cast<char>(cacheBuffer_[i]);
                if (symbol != kNBGMagic[i]) {
                    cacheBuffer_.clear();
                }
            }
        }

        cacheState_ = cacheBuffer_.empty() ? CacheState::EMPTY : CacheState::LOADED;
    }

    finished_ = true;
    return ANEURALNETWORKS_NO_ERROR;
}

int Compilation::writeToCache(const uint8_t* data, size_t size) {
    if (cacheState_ == CacheState::DISABLED) {
        LOGE("Compilation::writeToCache cache is disabled");
        return ANEURALNETWORKS_BAD_STATE;
    }

    cacheBuffer_.resize(size);
    std::copy_n(data, size, cacheBuffer_.data());

    lseek(cacheFd_, 0, SEEK_SET);
    ssize_t writeSize = 0;
    size_t bufferOffset = 0;
    do {
        writeSize = write(cacheFd_, cacheBuffer_.data() + bufferOffset, size - bufferOffset);
        bufferOffset += writeSize;
    } while (writeSize > 0);

    if (writeSize < 0) {
        LOGE("Compilation::writeToCache failed to write cache file: %s (%d)", strerror(errno),
             errno);
        return ANEURALNETWORKS_BAD_DATA;
    }

    cacheState_ = CacheState::LOADED;
    return ANEURALNETWORKS_NO_ERROR;
}

}  // namespace vsi::android::sl