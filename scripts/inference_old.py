    # def __get_distance(self, tensor1, tensor2):
    #     img1 = get_image_tensor(img1_path)
    #     img1 = img1.cuda()
    #     out1 = self.model.forward_once(img1)

    #     img2 = get_image_tensor(img2_path)
    #     img2 = img2.cuda()
    #     out2 = self.model.forward_once(img2)

    #     distance = nn.functional.pairwise_distance(out1, out2)
    #     distance = distance.detach().cpu().numpy()[0]
    #     return distance
    # def predict(self, test_img_path, k, thresh):
    #     # get the distances
    #     distances = []
    #     for gt_img in self.dataset:
    #         if test_img_path == gt_img:
    #             continue
    #         distance = self.__get_distance(gt_img, test_img_path)
    #         distances.append(distance)
        
    #     smallest_indices = np.argpartition(distances, k)[:k]
    #     # apply threshold
    #     if distances[smallest_indices[-1]] > thresh:
    #         prediction = "unknown"
        
    #     else:
    #         closest_imgs = np.array(self.dataset)[smallest_indices]
    #         labels = [get_label(img_path) for img_path in closest_imgs]
    #         prediction = max(set(labels), key = labels.count)
    #     return prediction
    
# def load_data(dataset_path):
#     dataset = []
#     labels = []
#     for dir in os.listdir(dataset_path):
#         labels.append(dir)
#         for img_str in os.listdir(os.path.join(dataset_path, dir)):
#             img_path = os.path.join(dataset_path, dir, img_str)
#             dataset.append(img_path)
#     return dataset, labels

# def get_label(img_path):
#     img_str = os.path.basename(img_path)
#     img_str = os.path.splitext(img_str)[0]
#     img_str = img_str[:4]
#     # img_str = img_str.split("_", 1)[0]
#     return img_str