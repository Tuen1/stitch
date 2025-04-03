# 多线程  无畸变无柱面，新增标定最佳缝合线(竖直线，基于竖直线周边区域融合)
import cv2
import time
import numpy as np

# 图像处理
class ImageProcessor:
    # 图像显示
    def cvshow(name, img):
        cv2.imshow(name, img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # 特征点检测与描述
    def sift_kp(image):
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        sift = cv2.SIFT_create()
        kp, des = sift.detectAndCompute(image, None)
        # kp_image = cv2.drawKeypoints(gray_image, kp, None)
        return kp, des

    # 特征点匹配
    def get_good_match(des1, des2, jindu):  # 筛选匹配程度较高的特征点对
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)  # des1为模板图，des2为匹配图
        matches = sorted(matches, key=lambda x: x[0].distance / x[1].distance)
        goodmatch = []
        for m, n in matches:
            if m.distance < jindu * n.distance:  # 筛选程度的不同对结果的影响很大，但不是一味的越小越好，也不是越大越好，标定时根据不同的情况选择。
                goodmatch.append(m)
        return goodmatch

    # 特征点匹配结果可视化
    def drawMatches(imageA, imageB, kpsA, kpsB, matches, status):  # 初始化可视化图片，将A、B图的匹配特征点连接到一起

        (hA, wA) = imageA.shape[:2]
        (hB, wB) = imageB.shape[:2]
        vis = np.zeros((max(hA, hB), wA + wB, 3), dtype="uint8")
        vis[0:hA, 0:wA] = imageA
        vis[0:hB, wA:] = imageB

        # 联合遍历，画出匹配对
        for ((trainIdx, queryIdx), s) in zip(matches, status):
            # 当点对匹配成功时，画到可视化图上
            if s == 1:
                # 画出匹配对
                ptA = (int(kpsA[queryIdx][0]), int(kpsA[queryIdx][1]))
                ptB = (int(kpsB[trainIdx][0]) + wA, int(kpsB[trainIdx][1]))
                cv2.line(vis, ptA, ptB, (0, 255, 0), 1)

        # 返回可视化结果
        return vis

    # 柱面投影
    def cylinder_projection(img, f):  # 柱面投影并标定出投影矩阵
        rows, cols = img.shape[:2]
        output_cols = int(2 * f * np.arctan(0.5 * cols / f))
        output = np.zeros((rows, output_cols, 3), dtype=np.uint8)

        # 初始化一个矩阵，用于存储坐标映射
        transform_matrix = np.full((rows, output_cols, 2), -1, dtype=np.float32)  # 用 (-1, -1) 表示无效映射

        for i in range(rows):
            for j in range(output_cols):
                # 计算圆柱坐标 (x, y)
                x = int(f * np.tan((j - 0.5 * output_cols) / f) + 0.5 * cols)
                y = int((i - 0.5 * rows) * np.sqrt((x - 0.5 * cols) ** 2 + f ** 2) / f + 0.5 * rows)

                # 检查计算出的坐标是否在原图像范围内
                if 0 <= x < cols and 0 <= y < rows:
                    output[i, j] = img[y, x]
                    transform_matrix[i, j] = [x, y]  # 在变换矩阵中存储映射的坐标
                else:
                    output[i, j] = [0, 0, 0]  # 设置图像边界之外的区域为黑色

        return output, transform_matrix

    # 预计算柱面投影
    def precompute_cylinder_projection(transform_matrix, img_shape):
        """
        预计算有效映射的布尔掩码和目标索引。

        Args:
            transform_matrix: 变换矩阵，形状为 (rows, cols, 2)。
            img_shape: 输入图像的形状 (height, width, channels)。

        Returns:
            valid_mask: 有效映射的布尔掩码。
            target_indices: 目标像素的 1D 索引。
            source_coords: 源图像的有效坐标数组，形状为 (N, 2)。
        """
        rows, output_cols = transform_matrix.shape[:2]

        # 计算有效映射的布尔掩码
        valid_mask = (
                (transform_matrix[:, :, 0] >= 0) &
                (transform_matrix[:, :, 1] >= 0) &
                (transform_matrix[:, :, 0] < img_shape[1]) &
                (transform_matrix[:, :, 1] < img_shape[0])
        )

        # 使用布尔掩码提取源图像的有效坐标
        source_coords = transform_matrix[valid_mask].astype(int)

        # 计算目标坐标的 1D 索引
        target_indices = np.flatnonzero(valid_mask)

        return valid_mask, target_indices, source_coords

    # 应用柱面投影
    def apply_cylinder_projection(img, rows, output_cols, valid_mask, target_indices, source_coords):
        """
        将输入图像像素映射到目标图像（实时计算部分）。

        Args:
            img: 输入图像，形状为 (height, width, channels)。
            rows: 目标图像的高度。
            output_cols: 目标图像的宽度。
            valid_mask: 有效映射的布尔掩码。
            target_indices: 目标像素的 1D 索引。
            source_coords: 源图像的有效坐标数组，形状为 (N, 2)。

        Returns:
            output: 经过柱面投影的目标图像，形状为 (rows, output_cols, 3)。
        """
        # 初始化输出图像
        output = np.zeros((rows, output_cols, 3), dtype=np.uint8)

        # 将源图像的有效像素值映射到目标图像
        output.reshape(-1, 3)[target_indices] = img[source_coords[:, 1], source_coords[:, 0]]

        return output

    # 查找左右边界的最小和最大Y值（使用掩膜优化）
    def find_left_right_boundaries(cyl_img):
        rows, cols = cyl_img.shape[:2]

        # 获取左边界（第一列）的掩膜，判断是否为非黑色像素
        left_col_mask = np.any(cyl_img[:, 0] != [0, 0, 0], axis=1)
        # 获取右边界（最后一列）的掩膜，判断是否为非黑色像素
        right_col_mask = np.any(cyl_img[:, cols - 1] != [0, 0, 0], axis=1)

        # 使用掩膜找到非黑色像素的行索引
        left_nonzero_rows = np.nonzero(left_col_mask)[0]
        right_nonzero_rows = np.nonzero(right_col_mask)[0]

        # 如果找到有效区域，则取最小和最大Y值，否则返回默认值
        if len(left_nonzero_rows) > 0:
            min_y_left, max_y_left = left_nonzero_rows[0], left_nonzero_rows[-1]
        else:
            min_y_left, max_y_left = rows, -1

        if len(right_nonzero_rows) > 0:
            min_y_right, max_y_right = right_nonzero_rows[0], right_nonzero_rows[-1]
        else:
            min_y_right, max_y_right = rows, -1

        return (min_y_left, max_y_left), (min_y_right, max_y_right)

# 拼接
class Stitcher:

    # 获取重叠区域掩码
    def get_overlap_region_mask(imA, imB):  # 获取重叠区域的mask
        """
        Given two images of the save size, get their overlapping region and
        convert this region to a mask array.
        """
        overlap = cv2.bitwise_and(imA, imB)  # 获取重叠区域  获取两张图片像素值都不为0的区域 即为重叠区域
        mask = Stitcher.get_mask(overlap)  # 获取重叠区域的mask
        # mask = cv2.dilate(mask, np.ones((2, 2), np.uint8), iterations=2)  #去噪
        return mask

    # 获取图像掩码
    def get_mask(img):  # 灰度二值化后得到图片的MASK
        """
        Convert an image to a mask array.
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 灰度化
        ret, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)  # 二值化      灰度二值化后得到图片的MASK
        return mask

    # 计算强度值
    def optimal_seam_rule_value(I1, I2, mask):  # 计算强度值
        I1 = np.float32(
            cv2.cvtColor(I1, cv2.COLOR_BGR2GRAY))  # 图像默认是unit8，值只能取0-255，转为np.float32后，值可以随意取，这样E才能取到实际的值而不是被限制在255内
        I2 = np.float32(cv2.cvtColor(I2, cv2.COLOR_BGR2GRAY))

        Sx = np.array([[-2, 0, 2], [-1, 0, 1], [-2, 0, 2]])
        Sy = np.array([[-2, -1, -2], [0, 0, 0], [2, 1, 2]])

        I1_Sx = cv2.filter2D(I1, -1, Sx)
        I1_Sy = cv2.filter2D(I1, -1, Sy)
        I2_Sx = cv2.filter2D(I2, -1, Sx)
        I2_Sy = cv2.filter2D(I2, -1, Sy)

        E_color = ((I1 - I2) ** 2) * 0.5  # 灰度差平方
        E_geometry = ((I1_Sx - I2_Sx) ** 2 + (I1_Sy - I2_Sy) ** 2)  # 结构差的平方
        E = E_color + E_geometry  # 权重 0.5 1

        E[mask == 0] = 1000000000  # 非重叠区域能量值都设置的很大，查找缝合线时就会自动规避
        return E

    # 查找最小竖直线
    def DP_find_seam(I1, I2):  # 找最小竖直线

        I1_yuan = I1.copy()
        I2_yuan = I2.copy()

        # I1 = cv2.resize(I1,None,fx=0.2,fy=0.2)  #把图片降采样
        # I2 = cv2.resize(I2,None,fx=0.2,fy=0.2)
        # scale_factor = 5

        overlapMask = Stitcher.get_overlap_region_mask(I1, I2)  # 确定重叠区域掩膜
        indices = np.where(overlapMask == 255)
        x_indices = indices[1]  # 提取x坐标的数组
        # 找到x坐标的最小值和最大值
        min_x = np.min(x_indices)
        max_x = np.max(x_indices)
        # print("min_x,max_x",min_x,max_x)

        I1 = I1[:, min_x:max_x]
        I2 = I2[:, min_x:max_x]
        overlapMask = overlapMask[:, min_x:max_x]

        start_time1 = time.perf_counter()
        E = Stitcher.optimal_seam_rule_value(I1, I2, overlapMask)  # 计算重叠区域的能量值，对于非重叠区域能量值取特别大，查找缝合线时就会自动规避，确保只在重叠区域查找缝合线
        end_time1 = time.perf_counter()
        # print(f"降采样后 计算E的时间: {end_time1 - start_time1} 秒")

        # 对每列的 E 值进行求和
        column_sums = np.sum(E, axis=0)  # 计算每列的 E 总值

        # 找到 E 总值最小的列坐标
        min_column = np.argmin(column_sums)  # 找到总和最小的列的索引
        # X = (min_column+min_x)*scale_factor
        X = (min_column + min_x)
        X = [X] * I1.shape[0]
        return X  # 返回seam_paths缝合线路径(只包含列信息)

    # 混合函数
    def blend(img1, img2, seam, transition_width):
        rows, cols = img1.shape[:2]

        # 创建权重矩阵
        weights = np.zeros((rows, cols), dtype=np.float32)
        for i in range(len(seam)):
            weights[i, 0:seam[i]] = 1

        # 获取重叠区域掩码
        overlap = Stitcher.get_overlap_region_mask(img1, img2)

        # 获取所有的位置
        blank_pixels = np.argwhere(overlap != 0)

        # 使用集合快速检查像素是否在blank_pixels中
        blank_pixels_set = set(map(tuple, blank_pixels))

        for y in range(rows):  # 使用行号作为y，seam中的值作为x
            x = seam[y]
            for offset in range(-transition_width // 2, transition_width // 2 + 1):  # 包含区间的右端点
                pos = x + offset
                if (y, pos) in blank_pixels_set:
                    # 线性权重函数
                    weights[y, pos] = (1 - offset / (transition_width // 2)) / 2

        # 扩展权重到通道维度
        weights = np.dstack([weights] * img1.shape[2])

        return weights


    # 全景拼接标定
    def siftimg_rightlignment(img_left, img_mid, img_right, chonhelv, L_jindu, R_jindu, pianyiliang):
        # 右图
        img_right = cv2.resize(img_right, None, fx=1, fy=1)  # 原始图片缩放过小，可能会导致图片透视变换时失真
        # 保证两张图一样大
        img_left = cv2.resize(img_left, (img_right.shape[1], img_right.shape[0]))
        img_mid = cv2.resize(img_mid, (img_right.shape[1], img_right.shape[0]))

        f = img_left.shape[1] * 0.7  # 根据图像宽度选择焦距

        # 对图像进行柱面投影变换

        img_left, transform_matrix = ImageProcessor.cylinder_projection(img_left, f)
        # img_right =  apply_cylinder_projection_optimized_v4(img_right, transform_matrix)
        # 预计算有效掩码和索引
        rows, output_cols = transform_matrix.shape[:2]
        # 预处理：计算有效掩码和坐标
        valid_mask, target_indices, source_coords = ImageProcessor.precompute_cylinder_projection(transform_matrix, img_right.shape)
        # 柱面投影实时计算：将输入图像像素映射到目标图像
        start_time1 = time.perf_counter()
        img_mid = ImageProcessor.apply_cylinder_projection(img_mid, rows, output_cols, valid_mask, target_indices, source_coords)
        img_right = ImageProcessor.apply_cylinder_projection(img_right, rows, output_cols, valid_mask, target_indices, source_coords)
        end_time1 = time.perf_counter()
        print(f"柱面投影时间: {end_time1 - start_time1} 秒")
        # #亮度初平衡
        w = img_left.shape[1]
        W = int(w * chonhelv)

        img_left = ColorBlance.colorblance_left(img_left, img_mid, W)
        img_right = ColorBlance.colorblance_right(img_mid, img_right, W)

        # 查找投影后图像中有效区域左右边界的Ymax Ymin  此时三图尺寸相同，柱面投影的焦距相同，所以有效区域尺寸也相同
        (min_y_left1, max_y_left1), (min_y_right1, max_y_right1) = ImageProcessor.find_left_right_boundaries(img_mid)
        # print((min_y_left1, max_y_left1), (min_y_right1, max_y_right1))
        y_x_1 = min(max_y_left1, max_y_right1)
        y_s_1 = max(min_y_left1, min_y_right1)

        difference = y_s_1 - y_x_1
        remainder = difference % 4
        if remainder != 0:
            y_s_1 += (4 - remainder)  # 调整 y_s_1 使得差值能被4整除

        img_mid = img_mid[y_s_1:y_x_1, :]  # 保留最大内接矩形，只裁剪高度，不裁剪宽度

        canvas = np.zeros((img_mid.shape[0], img_left.shape[1] + img_mid.shape[1], 3),
                          dtype=img_mid.dtype)  # 创建一个空的大画布，调整中图位置，避免后续左图变换时超出边界
        np.copyto(canvas[:, img_left.shape[1]:], img_mid)  # canvas为调整位置后的中图

        kp1, des1 = ImageProcessor.sift_kp(img_left)  # 源点
        kp2, des2 = ImageProcessor.sift_kp(canvas)  # 目标点
        goodMatch1 = ImageProcessor.get_good_match(des1, des2, L_jindu)

        if len(goodMatch1) > 4:
            # 获取匹配对的点坐标
            ptsA = np.float32([kp1[m.queryIdx].pt for m in goodMatch1]).reshape(-1, 1, 2)  # 左
            ptsB = np.float32([kp2[m.trainIdx].pt for m in goodMatch1]).reshape(-1, 1, 2)  # 中
            ransacReprojThreshold = 4
            H_left, status = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, ransacReprojThreshold)
            #  该函数的作用就是先用RANSAC选择最优的四组配对点，再计算H矩阵。H为3*3矩阵

            # 将图片左进行视角变换，result是基于特征点变换后的图片
            result = cv2.warpPerspective(img_left, H_left, (canvas.shape[1], canvas.shape[0]))  # 左柱面透视变换后再裁成与中图同高
            ImageProcessor.cvshow('result_medium', result)

            # 图像拼接
            ImageProcessor.cvshow('CANVAS', canvas)

            # M,t_input = pinyipeizun_zero(canvas,result)
            # result = cv2.warpAffine(result, M, (result.shape[1], result.shape[0]))

            X_left = Stitcher.DP_find_seam(canvas, result)  # 在重叠区域中找拼接线   canvas指中图，result指变换后的左图
            # result[:, 0:X] = canvas[:, 0:X]
            # for i in range(len(X)):
            # result[i, 0:X[i]+1] = canvas[i, 0:X[i]+1]
            # for i in range(len(X_left)):
            #       X_left[i] = X_left[i]+30
            weights_left = Stitcher.blend(result, canvas, X_left, transition_width=80)
            result_left = (weights_left * result + (1 - weights_left) * canvas).astype(np.uint8)

            ImageProcessor.cvshow('stitch_left_img', result_left)

        kp1, des1 = ImageProcessor.sift_kp(img_right)  # 源点
        kp2, des2 = ImageProcessor.sift_kp(img_mid)  # 目标点
        goodMatch2 = ImageProcessor.get_good_match(des1, des2, R_jindu)
        # 当筛选项的匹配对大于4对时：计算视角变换矩阵
        if len(goodMatch2) > 4:
            # 获取匹配对的点坐标
            ptsA = np.float32([kp1[m.queryIdx].pt for m in goodMatch2]).reshape(-1, 1, 2)
            ptsB = np.float32([kp2[m.trainIdx].pt for m in goodMatch2]).reshape(-1, 1, 2)
            ransacReprojThreshold = 4
            H_right, status = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, ransacReprojThreshold)
            #  该函数的作用就是先用RANSAC选择最优的四组配对点，再计算H矩阵。H为3*3矩阵

            # 将图片右进行视角变换，result是变换后图片
            result = cv2.warpPerspective(img_right, H_right,
                                         (img_right.shape[1] + img_mid.shape[1], img_mid.shape[0]))  # 右柱面透视变换后再裁成与中图同高
            ImageProcessor.cvshow('result_medium', result)

            canvas = np.zeros((result.shape[0], result.shape[1], 3), dtype=result.dtype)  # 创建一个空的大画布
            np.copyto(canvas[:, :img_mid.shape[1]], img_mid)

            # M,t_input = pinyipeizun_zero(canvas,result)
            # result = cv2.warpAffine(result, M, (result.shape[1], result.shape[0]))

            X_right = Stitcher.DP_find_seam(canvas, result)  # 在重叠区域中找拼接线   canvas指中图，result指变换后的右图
            # result[:, 0:X] = canvas[:, 0:X]
            # for i in range(len(X)):
            # result[i, 0:X[i]+1] = canvas[i, 0:X[i]+1]

            weights_right = Stitcher.blend(canvas, result, X_right, transition_width=50)
            result_right = (weights_right * canvas + (1 - weights_right) * result).astype(np.uint8)

            ImageProcessor.cvshow('stitch_right_img', result_right)

            # return H,X,rows, output_cols, valid_mask, target_indices, source_coords,y_s_1,y_x_1,W,M,t_input,weights
            # return H,X,W
        a = result_left.shape[1] - int(img_mid.shape[1] / 2) + pianyiliang
        b = int(img_mid.shape[1] / 2) + pianyiliang
        start_time1 = time.perf_counter()
        L_IMG = result_left[:, :a]
        R_IMG = result_right[:, b:]
        ALL_IMG = np.zeros((L_IMG.shape[0], L_IMG.shape[1] + R_IMG.shape[1], 3),
                           dtype=L_IMG.dtype)  # 创建一个空的大画布，调整中图位置，避免后续左图变换时超出边界
        np.copyto(ALL_IMG[:, :L_IMG.shape[1]], L_IMG)  # canvas为调整位置后的中图
        np.copyto(ALL_IMG[:, L_IMG.shape[1]:], R_IMG)  # canvas为调整位置后的中图
        end_time1 = time.perf_counter()
        print(f"最终拼接时间: {end_time1 - start_time1} 秒")
        # ALL_IMG = cv2.resize(ALL_IMG,None,fx=0.8,fy=0.8)
        cv2.namedWindow('over_img', flags=cv2.WINDOW_FREERATIO)
        ImageProcessor.cvshow('over_img', ALL_IMG[:, 500:ALL_IMG.shape[1] - 500])

        # cv2.imwrite("C:/Users/HASEE/345.jpg", ALL_IMG)  #将取得的每一帧图片保存
        return H_left, H_right, rows, output_cols, valid_mask, target_indices, source_coords, y_s_1, y_x_1, W, weights_left, weights_right, a, b

# 亮度平衡
class ColorBlance:
    # 亮度平衡（右侧图像）
    def colorblance_right(img_L, img_R, W):
        # 亮度初调整
        # 将图像转换为灰度图
        # 计算亮度差
        b, g, r, _ = cv2.mean(img_L[:, img_L.shape[1] - W:])
        B, G, R, _ = cv2.mean(img_R[:, :W])
        brightness_increase = (
            np.clip(int(b - B), -50, 50),  # 蓝色通道差值
            np.clip(int(g - G), -50, 50),  # 绿色通道差值
            np.clip(int(r - R), -50, 50),  # 红色通道差值
        )

        # 分别解包 BGR 的亮度差值
        delta_b, delta_g, delta_r = brightness_increase

        # 创建一个空的调整矩阵
        adjustment = np.zeros_like(img_R, dtype=np.uint8)

        # 分别填充对应通道的调整值
        adjustment[:, :, 0] = abs(delta_b)  # 蓝色通道
        adjustment[:, :, 1] = abs(delta_g)  # 绿色通道
        adjustment[:, :, 2] = abs(delta_r)  # 红色通道

        # 调整 img_right 的亮度
        img_right_adjusted = img_R.copy()
        if delta_b >= 0:
            img_right_adjusted[:, :, 0] = cv2.add(img_R[:, :, 0], adjustment[:, :, 0])
        else:
            img_right_adjusted[:, :, 0] = cv2.subtract(img_R[:, :, 0], adjustment[:, :, 0])

        if delta_g >= 0:
            img_right_adjusted[:, :, 1] = cv2.add(img_R[:, :, 1], adjustment[:, :, 1])
        else:
            img_right_adjusted[:, :, 1] = cv2.subtract(img_R[:, :, 1], adjustment[:, :, 1])

        if delta_r >= 0:
            img_right_adjusted[:, :, 2] = cv2.add(img_R[:, :, 2], adjustment[:, :, 2])
        else:
            img_right_adjusted[:, :, 2] = cv2.subtract(img_R[:, :, 2], adjustment[:, :, 2])

        # 更新 img_right
        img_R = img_right_adjusted
        return img_R

    # 亮度平衡（左侧图像）
    def colorblance_left(img_L, img_R, W):
        # 亮度初调整
        # 将图像转换为灰度图
        # 计算亮度差
        b, g, r, _ = cv2.mean(img_L[:, img_L.shape[1] - W:])
        B, G, R, _ = cv2.mean(img_R[:, :W])
        brightness_increase = (
            np.clip(int(b - B), -50, 50),  # 蓝色通道差值
            np.clip(int(g - G), -50, 50),  # 绿色通道差值
            np.clip(int(r - R), -50, 50),  # 红色通道差值
        )

        # 分别解包 BGR 的亮度差值
        delta_b, delta_g, delta_r = brightness_increase

        # 创建一个空的调整矩阵
        adjustment = np.zeros_like(img_L, dtype=np.uint8)

        # 分别填充对应通道的调整值
        adjustment[:, :, 0] = abs(delta_b)  # 蓝色通道
        adjustment[:, :, 1] = abs(delta_g)  # 绿色通道
        adjustment[:, :, 2] = abs(delta_r)  # 红色通道

        # 调整 img_left 的亮度
        img_left_adjusted = img_L.copy()
        if delta_b >= 0:
            img_left_adjusted[:, :, 0] = cv2.subtract(img_L[:, :, 0], adjustment[:, :, 0])
        else:
            img_left_adjusted[:, :, 0] = cv2.add(img_L[:, :, 0], adjustment[:, :, 0])

        if delta_g >= 0:
            img_left_adjusted[:, :, 1] = cv2.subtract(img_L[:, :, 1], adjustment[:, :, 1])
        else:
            img_left_adjusted[:, :, 1] = cv2.add(img_L[:, :, 1], adjustment[:, :, 1])

        if delta_r >= 0:
            img_left_adjusted[:, :, 2] = cv2.subtract(img_L[:, :, 2], adjustment[:, :, 2])
        else:
            img_left_adjusted[:, :, 2] = cv2.add(img_L[:, :, 2], adjustment[:, :, 2])

        # 更新 img_right
        img_L = img_left_adjusted
        return img_L


def printMatrixSize(matrix):
    if isinstance(matrix, np.ndarray):  # 检查是否为 NumPy 数组
        print(f"矩阵的大小为 {matrix.shape}")
    else:
        print("矩阵为空或类型不支持")

def printArrayStats(array):
    if isinstance(array, np.ndarray):
        print(f"数组的形状: {array.shape}")
        print(f"最小值: {np.min(array)}")
        print(f"最大值: {np.max(array)}")
        print(f"非零值的数量: {np.count_nonzero(array)}")
    else:
        print("输入不是 NumPy 数组")
if __name__ == '__main__':
    # 以左图为基准，对右边的图形做变换
    img_right = cv2.imread(r"C:\Users\74002\Desktop\now\right.jpg")
    img_left = cv2.imread(r"C:\Users\74002\Desktop\now\left.jpg")
    img_mid = cv2.imread(r"C:\Users\74002\Desktop\now\mid.jpg")

    # 把图片拼接成全景图，标定求出拼接参数

    chonhelv = 0.6  # 两图间的重合率百分比
    L_jindu = 0.9  # 两图的get_good_match的阈值精度可独立设置，用于分别找到最合适的H
    R_jindu = 0.9
    pianyiliang = 0  # 最终拼接线的偏移量，+右移 -左移

    H_left, H_right, rows, output_cols, valid_mask, target_indices, source_coords, y_s_1, y_x_1, W, weights_left, weights_right, a, b = Stitcher.siftimg_rightlignment(
        img_left, img_mid, img_right, chonhelv, L_jindu, R_jindu, pianyiliang)
    assert len(target_indices) == len(source_coords), "target_indices and source_coords sizes do not match"
    np.savez("calibration_data.npz",
             H_left=H_left,
             H_right=H_right,
             rows=rows,
             output_cols=output_cols,
             valid_mask=valid_mask,
             # target_indices=target_indices,
             target_indices=target_indices,
             source_coords=source_coords,
             y_s_1=y_s_1,
             y_x_1=y_x_1,
             W=W,
             weights_left=weights_left,
             weights_right=weights_right,
             a=a,
             b=b)
    # data = {
    #     "rows" : rows,
    #     "output_cols" : output_cols,
    #     "y_x_1" : y_x_1,
    #     "W" : W,
    #     "a" : a,
    #     "b" : b
    # }
    # def int64_to_int(o):
    #     if isinstance(o, np.int64):
    #         return int(o)
    #     raise TypeError
    # json_str = json.dumps(data, default=int64_to_int)
    # # 将JSON字符串保存到文件
    # with open('data.json', 'w') as json_file:
    #     json.dump(data, json_file, default=int64_to_int)


    # np.save("H_left.npy", H_left)
    data = np.load("calibration_data.npz")

    # 访问各个变量
    H_left = data['H_left']
    H_right = data['H_right']
    rows = data['rows']
    output_cols = data['output_cols']
    valid_mask = data['valid_mask']
    target_indices = data['target_indices']
    # Python端检查数据类型
    print("-----------:",target_indices.dtype)  # 应输出int64或类似

    source_coords = data['source_coords']
    y_s_1 = data['y_s_1']
    y_x_1 = data['y_x_1']
    W = data['W']
    weights_left = data['weights_left']
    weights_right = data['weights_right']
    print("-----------:", weights_right.dtype)  # 应输出int64或类似
    printMatrixSize(weights_right)
    a = data['a']
    b = data['b']

    # 打印读取的数据
    print("H_left:\n", H_left)
    print("H_right:\n", H_right)
    print("rows:", rows)
    print("output_cols:", output_cols)
    print("valid_mask:\n", valid_mask)
    print("target_indices:\n", target_indices)
    print("target_indices_len:",len(target_indices))
    printArrayStats(target_indices)
    print("前几个值:", target_indices[:10])  # 打印前 10 个值
    print("最后几个值:", target_indices[-10:])  # 打印最后 10 个值
    is_continuous = np.all(np.diff(target_indices) == 1)
    print("数组是否连续:", is_continuous)


    print("source_coords:\n", source_coords)
    print("source_coords_len:", len(source_coords))
    print("y_s_1:", y_s_1)
    print("y_x_1:", y_x_1)
    print("W:", W)
    print("weights_left:\n", weights_left)
    print("weights_right:\n", weights_right)
    print("a:", a)
    print("b:", b)

    data.close()

