import numpy as np
from nilearn import image



atlases = ['/data/CAN/brainnetome.nii','/data/CAN/Yeo-7.nii.gz']

at1 = image.load_img(atlases[0])
at2 = image.load_img(atlases[1])
atlas1 = at1.get_fdata()
atlas2 = at2.get_fdata()

# Get ROI numerical values for both atlases
labs1 = np.unique(atlas1)
labs2 = np.unique(atlas2)

# Create ndarray of zeros to contain Dice Coefficients  注意，由于0代表背景，因此要size-1
Dice = np.zeros((labs1.size - 1, labs2.size - 1))


for i in range(1, len(labs1)):    #剔除背景编号0
    val1 = labs1[i]
    for j in range(1, len(labs2)):  #剔除背景编号0
        val2 = labs2[j]
        dice = np.sum(atlas1[atlas2 == val2] == val1) / np.sum(atlas1[atlas1 == val1] == val1)

        # Store in ndarray
        Dice[int(i)-1][int(j)-1] = float(dice)


        # Check for false Dice Coefficients and return what ROIs caused the issue
        if dice > 1 or dice < 0:
            raise ValueError(
                f"Dice coefficient is greater than 1 or less than 0 ({dice}) at atlas1: {val1}, atlas2: {val2}")

# # Save Dice map to csv file, comma delimited
# np.savetxt('./yeo-7.csv', Dice, delimiter=",")
# 设置阈值，得到初始化的社区分配结果
threshold = 0.8
community_num = 7   #具有七个社区
cortex_node_num = 210 #使用的脑图谱包含210个皮层节点
community_label = {}
for c in range(community_num):   #初始化社区标签
    community_label[c] = []

for node in range(cortex_node_num):
    max_dice = (Dice[node]).max()
    max_dice_idx = (Dice[node]).argmax()
    if max_dice > threshold:
        community_label[max_dice_idx].append(node)

for c in range(community_num):
    print(community_label[c])




