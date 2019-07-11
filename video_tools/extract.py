import cv2
import numpy as np
import matplotlib.pyplot as plt
import imutils
import copy


class extractor(object):
    def version(self):
        print(self._version)
        
        
    def __init__(self, BLUR=3, CANNY_THRESH_1=10, CANNY_THRESH_2=200, MASK_DILATE_ITER=4, MASK_ERODE_ITER=4, max_area=224*224):
        self._version = 0.1
        #== Parameters =======================================================================
        self.BLUR = BLUR
        self.CANNY_THRESH_1 = CANNY_THRESH_1
        self.CANNY_THRESH_2 = CANNY_THRESH_2
        self.MASK_DILATE_ITER = MASK_DILATE_ITER
        self.MASK_ERODE_ITER = MASK_ERODE_ITER
        self.MASK_COLOR = (0.0,0.0,1.0) # In BGR format
        self.max_area = max_area


    def extract_positive(self, img):
        # img is a 3 channels RGB array
        # img_returned is a 4 hannel RGB array
        if img.shape[2] != 3:
            raise AssertionError('Input image needs to have 3 channels')
            
        #== Processing =======================================================================
        
        #-- Read image -----------------------------------------------------------------------
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)  # The input image is loaded with plt (RGB)

        #-- Edge detection -------------------------------------------------------------------
        # Returns a list containing info about the largest area contour
        edges = cv2.Canny(gray, self.CANNY_THRESH_1, self.CANNY_THRESH_2)
        edges = cv2.dilate(edges, None)  # Clean up edges
        edges = cv2.erode(edges, None)
        
        
        #-- Find contours in edges, sort by area ---------------------------------------------
        contour_info = []
        contours = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        contours = imutils.grab_contours(contours)

        for c in contours:
            contour_info.append((
                c,
                cv2.isContourConvex(c),
                cv2.contourArea(c),
            ))
        contour_info = sorted(contour_info, key=lambda c: c[2], reverse=True)
        max_contour = contour_info[0]
        
        # Filter out the contours that are bigger than the original crop
        if max_contour[2] > self.max_area:
            area_returned = 0
            img_returned = np.zeros((img.shape[0], img.shape[1], 4), dtype=np.uint8)
        else:
            #plt.figure()
            #plt.imshow(edges, cmap='gray')
            #plt.show()
            
            #-- Create empty mask, draw filled polygon on it corresponding to largest contour ----
            # Mask is black, polygon is white
            mask = np.zeros(edges.shape)
            #cv2.fillConvexPoly(mask, max_contour[0], (255))
            image = cv2.fillPoly(mask, max_contour[0], 255)  # Outputs a float64 array
            h, w = image.shape[:2]
            image = np.array(image, dtype=np.uint8)
            mask = np.zeros((h+2, w+2), np.uint8)
            _, mask, _, _ = cv2.floodFill(image.copy(), mask, (112, 112), 255)

            #-- Additional post processing to improve the mask quality--------------------------------------------------------
            # These param were found to be optimal for my application
            #mask = cv2.dilate(mask, None, iterations=self.MASK_DILATE_ITER)
            #mask = cv2.erode(mask, None, iterations=self.MASK_ERODE_ITER)
            #mask = cv2.GaussianBlur(mask, (self.BLUR, self.BLUR), 0)
            plt.figure()
            plt.imshow(mask, cmap='gray')
            plt.show()
            #print(mask.dtype, np.max(mask), np.min(mask))
            #assert 1==0

            img         = img.astype('float32') / 255.0                 #  for easy blending
            # split image into channels
            c_red, c_green, c_blue = cv2.split(img)

            # merge with mask got on one of a previous steps
            img_a = cv2.merge((c_red, c_green, c_blue, mask.astype('float32') / 255.0))
            
            # Convert back to RGB
            img_a *= 255
            img_a = img_a.astype(np.uint8)

            #img_a = np.array(img_a*255.0, dtype=np.uint8)
            # print("Extracted saved with range from {} to {}".format(np.min(img_a), np.max(img_a)))
            area_returned = max_contour[2]
            img_returned = img_a
            # Save to file
            #plt.imsave(output_path, img_a)
        
        return area_returned, img_returned


    def blend_with_negative(self, negative, positive, rotations=(0, 0), scaling=(1, 1)):
        # Rotation is taken from a normal distribution rotations=(center, std)
        if positive.shape[2] != 4:
            raise AssertionError('The positive image needs to have 4 channels')
        if negative.shape[2] != 3:
            raise AssertionError('The negative image needs to have 3 channels')
        # positive image has 4 channels RGB, negative only 3 RGB
        if np.max(positive) <= 1:
            raise AssertionError('The positive image needs to be RGBA uint8')
        if np.max(negative) <= 1:
            raise AssertionError('The negative image needs to be RGBA uint8')
        # Returns an RGB array
        
        # 1. Set positive's RGB brightness to negative's
        positive = self.adjust_brightness(negative, positive)
        # 2. Geometric transformations
        # 2.1 Rotate the positive image
        if rotations[1] > 0:
            angle = np.random.normal(rotations[0], rotations[1])
            # print(angle)
            positive = self.rotate_image(positive, angle)
        # 2.2 Zoom in on the image /!\ zoom out not supported (yet?)
        if scaling[1] > 0:
            zoom = np.random.normal(scaling[0], scaling[1])
            zoom = 1 if zoom <= 1 else zoom  # Limit zoom to zero
            row_c = positive.shape[0]//2
            col_c = positive.shape[1]//2
            positive = cv2.resize(positive, None, fx=zoom, fy=zoom, interpolation=cv2.INTER_CUBIC)
            positive = positive[int(row_c*(zoom-1)):int(row_c*(zoom+1)), int(col_c*(zoom-1)):int(col_c*(zoom+1))]
        # 3. Blend images 
        # 3.1 Convert to [0, 1] for easy blending
        positive = positive.astype('float32') / 255.0
        negative = negative.astype('float32') / 255.0
        # 3.2 Blending is alpha*positive + (1-alpha)*negative 
        alpha_stack = np.dstack([positive[:, :, 3]]*3)  # Select the alpha
        overlay = cv2.multiply(alpha_stack, positive[:, :, :3])
        underlay = cv2.multiply(1.0 - alpha_stack, negative)
        blended_image = cv2.add(overlay, underlay)
        # 3.3 Restore to [0 255]
        blended_image = np.array(blended_image*255, dtype=np.uint8)
        if blended_image.shape[2] != 3:
            raise AssertionError('The blended image does not have 3 channels')
        return blended_image
        
        
    def rotate_image(self, img_array, angle):
        # Takes an array as input and returns the rotated image around its center
        m = cv2.getRotationMatrix2D((img_array.shape[1]//2, img_array.shape[0]//2), angle, 1)
        return cv2.warpAffine(img_array, m, (img_array.shape[1], img_array.shape[0]))


    def adjust_brightness(self, reference, image_to_adjust):
        # Reference is RGB 3 channels
        # image_to_adjust is RGB 3 or 4 channels
        # Return is RGB 3 or 4 channels (depending on input)
        if reference.shape[2] != 3:
            raise AssertionError('The reference image needs to have 3 channels')
        if image_to_adjust.shape[2] != 3 and image_to_adjust.shape[2] != 4:
            raise AssertionError('The image_to_adjust image needs to have 3 or 4 channels')
        # Use a conversion to HSV to extract the brightness and adjust
        img = image_to_adjust.copy()
        img_hsv = cv2.cvtColor(img[:, :, :3], cv2.COLOR_RGB2HSV)
        reference_hsv = cv2.cvtColor(reference, cv2.COLOR_RGB2HSV)
        # print(np.mean(reference_hsv[2]), np.mean(img_hsv[2]))
        img_hsv[2] = reference_hsv[2]  # Copy brightness values from reference
        adjusted_image = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)
        if image_to_adjust.shape[2] == 4:  # Need to merge the alpha mask
            alpha = image_to_adjust[:, :, 3]  # Slicing makes it 2D
            alpha = alpha[..., np.newaxis]  # Add the mask back
            # print(alpha.shape)
            adjusted_image = np.concatenate((adjusted_image, alpha), axis=2)
        # print(np.mean(image_to_adjust), np.mean(adjusted_image))
        return adjusted_image
