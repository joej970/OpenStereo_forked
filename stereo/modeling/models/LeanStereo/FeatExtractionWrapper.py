

import torch
import torch.nn as nn

class feature_extraction(nn.Module):
    def __init__(self, feature_extractor, concat_feature=False, concat_feature_channel=12, cfgs = None):
        super(feature_extraction, self).__init__()
        self.concat_feature = concat_feature
        self.features = feature_extractor
        if self.concat_feature:
            self.lastconv = nn.Sequential(convbn(320, 128, 3, 1, 1, 1),
                                          nn.ReLU(inplace=True),
                                          nn.Conv2d(128, concat_feature_channel, kernel_size=1, padding=0, stride=1,
                                                    bias=False))

        # by default, do two separate passes (traditional way)
        # self.forward = self.forward_single_image
        self.forward = self.forward_left_right_one_by_one

        # check if batching requested (new proposal)
        if (cfgs is not None):
            should_concat = cfgs.get('CONCAT_LEFT_RIGHT', False) # if not supplied, then assume False
            if should_concat:
                concat_type = cfgs.get('CONCAT_LEFT_RIGHT_ALONG', None)
                if concat_type is None:
                    raise ValueError(f"Could not find MODEL.BACKBONE_CFGS.CONCAT_LEFT_RIGHT_ALONG parameter in yaml config file.")
                print(f"BACKBONE CONCAT LEFT-RIGHT is ENABLED. Type: {concat_type}")
                if concat_type == 'batch':
                    self.forward = self.forward_left_right_images_along_batch
                elif concat_type == 'horizontal':
                    self.forward = self.forward_left_right_images_horizontal
                elif concat_type == 'vertical':
                    self.forward = self.forward_left_right_images_vertical
                elif concat_type == 'multicut':
                    self.forward = self.forward_left_right_images_multicut
                    self.h_division = cfgs.get('H_DIVISION', 1)
                    self.w_division = cfgs.get('W_DIVISION', 2)
                    print(f"Multicut concat: H_DIVISION={self.h_division}, W_DIVISION={self.w_division}")
                    # if self.h_division is None or self.w_division is None:
                    #     raise ValueError(f"MODEL.BACKBONE_CFGS.H_DIV and W_DIV must be specified for multicut concatenation.")                    
                else:
                    raise NotImplementedError(f"Concat type '{concat_type}' is not implemented. Available: batch, horizontal, vertical, multicut")

    
    def forward_left_right_one_by_one(self, left, right):
        # first left
        features_left = self.features(left)
        if self.concat_feature:
            concat_feature_left = self.lastconv(features_left)

        # then right
        features_right = self.features(right)        
        if self.concat_feature:
            concat_feature_right = self.lastconv(self.features(right))

        # return together
        if self.concat_feature:
            return {"features": features_left, "concat_feature": concat_feature_left}, \
                     {"features": features_right, "concat_feature": concat_feature_right}
        else:
            return {"features": features_left}, {"features": features_right}
 

    def forward_left_right_images_along_batch(self, left, right):
        # concatenate left and right images along batch dimension
        batch_size, _, height, width = left.size()
        concat_images = torch.cat((left, right), dim=0)  # [2*B, C, H, W]
        features = self.features(concat_images)
        if not self.concat_feature:
            # split features back to left and right
            features_left = {"features": features[:batch_size]}
            features_right = {"features": features[batch_size:]}
            return features_left, features_right
        else:
            concat_feature = self.lastconv(features)
            concat_feature_left = concat_feature[:batch_size]
            concat_feature_right = concat_feature[batch_size:]
            return {"features": features[:batch_size], "concat_feature": concat_feature_left}, \
                   {"features": features[batch_size:], "concat_feature": concat_feature_right}
        
    def forward_left_right_images_horizontal(self, left, right):
        # concatenate left and right images along width dimension
        concat_images = torch.cat((left, right), dim=-1)  # [B, C, H, 2*W]
        features = self.features(concat_images)
        w = features.size(-1)
        w_divided = w // 2
        if not self.concat_feature:
            # split features back to left and right
            features_left  = {"features": features[..., :w_divided]}
            features_right = {"features": features[..., w_divided:]}
            return features_left, features_right
        else:
            concat_feature = self.lastconv(features)
            concat_feature_left  = concat_feature[..., :w_divided]
            concat_feature_right = concat_feature[..., w_divided:]
            return {"features": features[..., :w_divided], "concat_feature": concat_feature_left}, \
                   {"features": features[..., w_divided:], "concat_feature": concat_feature_right}

    def forward_left_right_images_vertical(self, left, right):
        # concatenate left and right images along height dimension
        concat_images = torch.cat((left, right), dim=-2)  # [B, C, 2*H, W]
        features = self.features(concat_images)
        h = features.size(-2)
        h_divided = h // 2
        if not self.concat_feature:
            # split features back to left and right
            features_left  = {"features": features[..., :h_divided, :]}
            features_right = {"features": features[..., h_divided:, :]}
            return features_left, features_right
        else:
            concat_feature = self.lastconv(features)
            concat_feature_left  = concat_feature[..., :h_divided, :]
            concat_feature_right = concat_feature[..., h_divided:, :]
            return {"features": features[..., :h_divided, :], "concat_feature": concat_feature_left}, \
                   {"features": features[..., h_divided:, :], "concat_feature": concat_feature_right}

    def forward_left_right_images_multicut(self, image_left, image_right):
        image_left_parts = []
        image_right_parts = []
        w_division = self.w_division
        h_division = self.h_division
        # after division, the new height and width need to be divisible by 32 (depending on the backbone)

        nr_of_parts = w_division * h_division  # 8
        h, w, b = image_left.size(-2), image_left.size(-1), image_left.size(0)
        h_divided = h // h_division
        w_divided = w // w_division
        assert h_divided % 32 == 0, f"Original image height {h} divided down to {h_divided} not divisible by 32."
        assert w_divided % 32 == 0, f"Original image width {w} divided down to {w_divided} not divisible by 32."

        for i in range(h_division):
            for j in range(w_division):
                image_left_parts.append(image_left[..., i*h_divided:(i+1)*h_divided, j*w_divided:(j+1)*w_divided])
                image_right_parts.append(image_right[..., i*h_divided:(i+1)*h_divided, j*w_divided:(j+1)*w_divided])

        left_stack = torch.cat(image_left_parts, dim=0)
        right_stack = torch.cat(image_right_parts, dim=0)

        combined = torch.cat((left_stack, right_stack), dim=0)

        combined_features = self.features(combined)

        # should be [b*2*8, c, h/8, w/8]
        combined_features_left = combined_features[:nr_of_parts*b, ...]
        combined_features_right = combined_features[nr_of_parts*b:, ...]
        
        left_feat_image = []
        right_feat_image = []
        
        for bi in range(b):

            left_v_sequence = []
            right_v_sequence = []
            for i in range(h_division): # along height

                left_h_sequence = []
                right_h_sequence = []
                for j in range(w_division): # along width

                    left_h_sequence.append(combined_features_left[i*b*w_division + j*b + bi, ...]) # append to horizontal sequence
                    right_h_sequence.append(combined_features_right[i*b*w_division + j*b + bi, ...])

                left_v_sequence.append(torch.cat(left_h_sequence, dim=-1)) # first concat along width (get full row), then append to vertical sequence
                right_v_sequence.append(torch.cat(right_h_sequence, dim=-1))

            left_feat_image.append(torch.cat(left_v_sequence, dim=-2)) # first concat along height (get full image), then append along batch dimension
            right_feat_image.append(torch.cat(right_v_sequence, dim=-2))

        features_left  = torch.stack(left_feat_image, dim=0)
        features_right = torch.stack(right_feat_image, dim=0)

        if not self.concat_feature:
            return {"features": features_left}, {"features": features_right}
        else:
            concat_feature = self.lastconv(combined_features)
            concat_feature_left = concat_feature[:nr_of_parts*b, ...]
            concat_feature_right = concat_feature[nr_of_parts*b:, ...]

            left_feat_image = []
            right_feat_image = []
            
            for bi in range(b):

                left_v_sequence = []
                right_v_sequence = []
                for i in range(h_division): # along height

                    left_h_sequence = []
                    right_h_sequence = []
                    for j in range(w_division): # along width

                        left_h_sequence.append(concat_feature_left[i*b*w_division + j*b + bi, ...]) # append to horizontal sequence
                        right_h_sequence.append(concat_feature_right[i*b*w_division + j*b + bi, ...])

                    left_v_sequence.append(torch.cat(left_h_sequence, dim=-1)) # first concat along width (get full row), then append to vertical sequence
                    right_v_sequence.append(torch.cat(right_h_sequence, dim=-1))

                left_feat_image.append(torch.cat(left_v_sequence, dim=-2)) # first concat along height (get full image), then append along batch dimension
                right_feat_image.append(torch.cat(right_v_sequence, dim=-2))

            concat_feature_left  = torch.stack(left_feat_image, dim=0)
            concat_feature_right = torch.stack(right_feat_image, dim=0)

            return {"features": features_left, "concat_feature": concat_feature_left}, \
                   {"features": features_right, "concat_feature": concat_feature_right}


    def forward_single_image(self, x):

        features = self.features(x)
        if not self.concat_feature:
            return {"features": features}
        else:
            concat_feature = self.lastconv(features)
            return {"features": features, "concat_feature": concat_feature}
        # return features