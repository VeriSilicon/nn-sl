
| op               | totoal | pass | failed | not support |
| -                | -      | -    | -      | -           |
| abs              | 4      | 3    | 1      | 0           |
| add              | 22     | 17   | 0      | 5           |
| avg_pool2d       | 160    | 140  | 0      | 20          |
| batch to space   | 64     | 52   | 0      | 12          |
| conv2d           | 854    | 830  | 0      | 24          |
| depth to space   | 80     | 80   | 0      | 0           |
| depthwise conv2d | 488    | 472  | 16     | 0           |
| dequantize       | 52     | 40   | 0      | 12          |
| div              | 23     | 20   | 0      | 3           |
| elu              | 15     | 15   | 0      | 0           |
| equal            | 31     | 31   | 0      | 0           |
| exp              | 3      | 3    | 0      | 0           |
| floor            | 6      | 6    | 0      | 0           |
| fully connected  | 106    | 96   | 0      | 10          |
| floor            | 6      | 6    | 0      | 3           |
| greater          | 31     | 31   | 0      | 0           |
| greater_equal    | 31     | 31   | 0      | 0           |
| grouped conv2d   | 464    | 464  | 0      | 0           |
| hard_swish       | 21     | 0    | 21     | 0           |
| l2_normalization | 274    | 274  | 0      | 0           |
| l2_pool2d        | 60     | 48   | 0      | 12          |
| less             | 31     | 31   | 0      | 0           |
| less_equal       | 31     | 31   | 0      | 0           |
| log              | 3      | 3    | 0      | 0           |
| logistic         | 25     | 20   | 0      | 5           |
| logical_not      | 1      | 1    | 0      | 0           |
| logical_and      | 2      | 2    | 0      | 0           |
| logical_or       | 2      | 2    | 0      | 0           |
| max_pool2d       | 132    | 112  | 0      | 20          |
| mean             | 46     | 24   | 0      | 22          |
| mul              | 26     | 21   | 0      | 5           |
| neg              | 4      | 4    | 0      | 0           |
| not_equal        | 31     | 31   | 0      | 0           |
| pad & pad_v2     | 128    | 56   | 8      | 64          |
| pow              | 24     | 24   | 0      | 0           |
| prelu            | 44     | 44   | 0      | 0           |
| quantize         | 38     | 32   | 0      | 6           |
| reduce_all       | 4      | 4    | 0      | 0           |
| reduce_any       | 4      | 4    | 0      | 0           |
| reduce_max       | 42     | 42   | 0      | 0           |
| reduce_min       | 42     | 42   | 0      | 0           |
| reduce_prod      | 26     | 26   | 0      | 0           |
| reduce_sum       | 26     | 26   | 0      | 0           |
| relu             | 29     | 24   | 0      | 5           |
| relu1            | 29     | 24   | 0      | 5           |
| relu6            | 29     | 24   | 0      | 5           |
| reshape          | 30     | 14   | 0      | 16          |
| rsqrt            | 11     | 11   | 0      | 0           |
| sin              | 3      | 3    | 0      | 0           |
| softmax          | 481    | 476  | 0      | 5           |
| space to depth   | 80     | 80   | 0      | 0           |
| space to batch   | 224    | 112  | 0      | 112         |
| sqrt             | 3      | 3    | 0      | 0           |
| sub              | 310    | 305  | 0      | 5           |
| tanh             | 15     | 10   | 0      | 5           |
| transpose conv2d | 604    | 500  | 84     | 20          |

**NOTE**: test result on vendor dev-kits with android 13