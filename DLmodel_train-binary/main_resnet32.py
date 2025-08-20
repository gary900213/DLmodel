import os
### ----------------------deadlift----------------------
# os.system("python train.py --GT_class 2 --F_type 3D")
# os.system("python train.py --GT_class 2 --SHAP abs")
# os.system("python train.py --GT_class 2 --SHAP avg_abs_min")
# os.system("python train.py --GT_class 2 --SHAP avg_min")

# os.system("python train.py --GT_class 3 --F_type 3D")
# os.system("python train.py --GT_class 3 --SHAP abs")
# os.system("python train.py --GT_class 3 --SHAP avg_abs_min")
# os.system("python train.py --GT_class 3 --SHAP avg_min")

# os.system("python train.py --GT_class 4 --F_type 3D")
# os.system("python train.py --GT_class 4 --SHAP abs")
# os.system("python train.py --GT_class 4 --SHAP avg_abs_min")
# os.system("python train.py --GT_class 4 --SHAP avg_min")

# os.system("python train.py --GT_class 5 --F_type 3D")
# os.system("python train.py --GT_class 5 --SHAP abs")
# os.system("python train.py --GT_class 5 --SHAP avg_abs_min")
# os.system("python train.py --GT_class 5 --SHAP avg_min")

# os.system("python train.py --GT_class 2 --F_type 2D")
# os.system("python train.py --GT_class 3 --F_type 2D")
# os.system("python train.py --GT_class 4 --F_type 2D")
# os.system("python train.py --GT_class 5 --F_type 2D")

### ----------------------benchpress----------------------
# os.system("python train.py --GT_class 0 --F_type 2D --sport benchpress --data BP_data_oringinal")
# os.system("python train.py --GT_class 1 --F_type 2D --sport benchpress --data BP_data_oringinal")
# os.system("python train.py --GT_class 2 --F_type 2D --sport benchpress --data BP_data_oringinal")
# os.system("python train.py --GT_class 3 --F_type 2D --sport benchpress --data BP_data_oringinal")
# os.system("python train.py --GT_class 4 --F_type 2D --sport benchpress --data BP_data_oringinal")

# os.system("python train.py --GT_class 0 --F_type 2D --sport benchpress --data BPdata --model BiLSTM")
# os.system("python train.py --GT_class 1 --F_type 2D --sport benchpress --data BPdata --model BiLSTM")
# os.system("python train.py --GT_class 2 --F_type 2D --sport benchpress --data BPdata --model BiLSTM")
# os.system("python train.py --GT_class 3 --F_type 2D --sport benchpress --data BPdata --model BiLSTM")
# os.system("python train.py --GT_class 4 --F_type 2D --sport benchpress --data BPPdata --model BiLSTM")


# os.system("python train.py --GT_class 2 --F_type 2D --sport deadlift --data 2D_traindata_Final --model BiLSTM")
# os.system("python train.py --GT_class 3 --F_type 2D --sport deadlift --data 2D_traindata_Final --model BiLSTM")
# os.system("python train.py --GT_class 4 --F_type 2D --sport deadlift --data 2D_traindata_Final --model BiLSTM")
# os.system("python train.py --GT_class 5 --F_type 2D --sport deadlift --data 2D_traindata_Final --model BiLSTM")

# os.system("python train.py --GT_class 2 --F_type 2D --sport deadlift --data 3D_Real_Final --model BiLSTM")
# os.system("python train.py --GT_class 3 --F_type 2D --sport deadlift --data 3D_Real_Final --model BiLSTM")
# os.system("python train.py --GT_class 4 --F_type 2D --sport deadlift --data 3D_Real_Final --model BiLSTM")
# os.system("python train.py --GT_class 5 --F_type 2D --sport deadlift --data 3D_Real_Final --model BiLSTM")

os.system("python train.py --GT_class 2 --F_type 2D --sport deadlift --data 2D_traindata_Final --model ResNet32")
os.system("python train.py --GT_class 3 --F_type 2D --sport deadlift --data 2D_traindata_Final --model ResNet32")
os.system("python train.py --GT_class 4 --F_type 2D --sport deadlift --data 2D_traindata_Final --model ResNet32")
os.system("python train.py --GT_class 5 --F_type 2D --sport deadlift --data 2D_traindata_Final --model ResNet32")

os.system("python train.py --GT_class 2 --F_type 2D --sport deadlift --data 3D_Real_Final --model ResNet32")
os.system("python train.py --GT_class 3 --F_type 2D --sport deadlift --data 3D_Real_Final --model ResNet32")
os.system("python train.py --GT_class 4 --F_type 2D --sport deadlift --data 3D_Real_Final --model ResNet32")
os.system("python train.py --GT_class 5 --F_type 2D --sport deadlift --data 3D_Real_Final --model ResNet32")

