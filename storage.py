        # one-hot-encode the mask
        # mask = one_hot_encode(mask, self.class_rgb_values).astype('float')
        # print(mask.max(), mask.min())



        # print(mask.shape)
        # print(mask.max(), mask.min())

        # # apply augmentations
        # if self.augmentation:
        #     sample = self.augmentation(image=image, mask=mask)
        #     image, mask = sample['image'], sample['mask']
        
        # # apply preprocessing
        # if self.preprocessing:
        #     # sample = self.preprocessing(image=image, mask=mask)
        #     # image, mask = sample['image'], sample['mask']
        #     image = self.preprocessing(image)
        #     mask = self.preprocessing(image)


# ENCODER = "resnet34"
# ENCODER_WEIGHTS = "imagenet"
# ACTIVATION = "sigmoid"


    # model = smp.DeepLabV3(encoder_name=ENCODER,
    #                       encoder_weights=ENCODER_WEIGHTS,
    #                       encoder_output_stride=8,
    #                       classes=1,
    #                       activation=ACTIVATION)
    # model.to(dp.DEVICE)


        # preds = torch.where(logits > 0.5, 1, 0)
    #     for i, metric in enumerate(METRICS):
    #         met = metric(preds, labels)
    #         # print(met.shape, met_vals.shape)
    #         # print(met.sum(dim=0))
    #         # print(met)
    #         met_vals[i] += met
    # met_vals /= len(dl)



# def train_val_loop(model, criterion, optimizer, dl, is_val):
#     if is_val:
#         model.eval()
#     else:
#         model.train()
#     running_loss = 0.0
#     # met_vals = torch.zeros(len(METRICS), dtype=torch.float32)
#     epoch_confusion = {"TP": 0, "FP": 0, "TN": 0, "FN": 0}
#     for data in tqdm(dl):
#         inputs = data[0].to(dp.DEVICE)
#         labels = data[1].to(dp.DEVICE)

#         logits:torch.Tensor = model(inputs)["out"]
#         logits = logits.view_as(labels)

#         loss = criterion(logits, labels)
#         if not(is_val):
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#         running_loss += loss.item()

#         conf = fuc.compute_confusion(logits, labels)
#         for key in epoch_confusion:
#             epoch_confusion[key] += conf[key]

#     running_loss /= len(dl)
#     # if is_val:
#     #     print("\tValid:")
#     # else:
#     #     print("\tTrain:")
#     # print(f'\t\tLoss: {running_loss:.3f}', end = "")
#     # for i in range(len(METRICS)):
#     #     print(f", {METRICS_NAMES[i]}: {met_vals[i]:.4f}", end="")
#     # print()
#     return running_loss, epoch_confusion




        # if tr_loss < 0.005:
        #     cnt_small += 1
        # else:
        #     cnt_small = 0
        # if cnt_small == 3:
        #     torch.save(model.state_dict(), os.path.join("saves", f"epoch_{epoch}_small.pth"))
        #     # plot = metrics_plot(train_losses, val_losses, train_f1s, val_f1s, train_accs, val_accs)
        #     metrics = dict(zip(METRICS_NAMES + ["Train_loss", "Val_loss"], val_mets.tolist() + [train_losses, val_losses]))
        #     plot = metrics_plot(**metrics)
        #     if not(os.path.exists("plots")):
        #         os.mkdir("plots")
        #     plot.savefig(os.path.join("plots", f"MetricsDeepLabv3_{epoch}Epoches_small.png"))
        #     break




# class RoadDataset(Dataset):

#     """Massachusetts Road Dataset. Read images, apply augmentation and preprocessing transformations.
    
#     Args:
#         images_dir (str): path to images folder
#         masks_dir (str): path to segmentation masks folder
#         class_rgb_values (list): RGB values of select classes to extract from segmentation mask
#         augmentation (albumentations.Compose): data transfromation pipeline 
#             (e.g. flip, scale, etc.)
#         preprocessing (albumentations.Compose): data preprocessing 
#             (e.g. noralization, shape manipulation, etc.)
    
#     """
    
#     def __init__(
#             self, 
#             images_dir, 
#             masks_dir, 
#             class_rgb_values=class_rgb_values, 
#             augmentation=None, 
#             preprocessing=None,
#         ):
        
#         self.image_paths = [os.path.join(images_dir, image_id) for image_id in sorted(os.listdir(images_dir))]
#         self.mask_paths = [os.path.join(masks_dir, image_id) for image_id in sorted(os.listdir(masks_dir))]
#         length = int(len(self.image_paths) * 0.2)
#         self.image_paths = self.image_paths[:length]
#         self.mask_paths = self.mask_paths[:length]
#         if len(self.image_paths) != len(self.mask_paths):
#             raise ValueError("Lengths of image list and mask list are unequal")

#         self.class_rgb_values = class_rgb_values
#         self.augmentation = augmentation
#         self.preprocessing = preprocessing
    
#     def __getitem__(self, i):
#         image = cv2.cvtColor(cv2.imread(self.image_paths[i]), cv2.COLOR_BGR2RGB)
#         mask = cv2.cvtColor(cv2.imread(self.mask_paths[i]), cv2.COLOR_BGR2GRAY)
#         # image = cv2.resize(image, (512, 512))
#         # mask = cv2.resize(mask, (512, 512))
#         if self.augmentation:
#             sample = self.augmentation(image=image, mask=mask)
#             image, mask = sample['image'], sample['mask']

#         image = torchvision.transforms.ToTensor()(image)
#         image = image.to(dtype=torch.float32)
#         mask = mask / 255
#         mask = torch.tensor(mask, dtype=torch.float32)
#         image = image.to(device=DEVICE)
#         mask = mask.to(device=DEVICE)
#         return image, mask
        
#     def __len__(self):
#         return len(self.image_paths)
    


# def IOU(logits, target):
#     '''
#     My IOU func
#     '''  
#     preds = torch.nn.functional.sigmoid(logits)
#     inter = (preds * target).sum()
#     union = (preds + target).sum() - inter
#     return inter / (union + 1e-7)



# def get_training_augmentation():
#     train_transform = [    
#         album.RandomCrop(height=TRAIN_CROP_SIZE, width=TRAIN_CROP_SIZE, pad_if_needed=True),
#         album.OneOf(
#             [
#                 album.HorizontalFlip(p=1),
#                 album.VerticalFlip(p=1),
#                 album.RandomRotate90(p=1),
#             ],
#             p=0.75,
#         ),
#         album.ToTensorV2()
#     ]
#     return album.Compose(train_transform)

# def get_validation_augmentation():   
#     # Add sufficient padding to ensure image is divisible by 32
#     test_transform = [
#         album.PadIfNeeded(min_height=1536, min_width=1536, border_mode=0),
#         album.ToTensorV2()
#     ]
#     return album.Compose(test_transform)



