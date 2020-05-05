docker run --rm -ti \
--name local_lenny \
-p 6006:6006 \
-v local_path_to_data:/storage/data \
-v local_path_to_code/Heterogeneous_CD/legacy:/storage/src \
llu025/lenny:gpu \
/bin/bash
