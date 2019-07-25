import cv2
import numpy as np
import matplotlib.pyplot as plt
import imutils
import copy


class extractor(object):
    def version(self):
        print(self._version)
        
        
    def __init__(self, BLUR=3, CANNY_THRESH_1=10, CANNY_THRESH_2=200, MASK_DILATE_ITER=4, MASK_ERODE_ITER=4):
        self._version = 0.1
        #== Parameters =======================================================================
        self.BLUR = BLUR
        self.CANNY_THRESH_1 = CANNY_THRESH_1
        self.CANNY_THRESH_2 = CANNY_THRESH_2
        self.MASK_DILATE_ITER = MASK_DILATE_ITER
        self.MASK_ERODE_ITER = MASK_ERODE_ITER
        self.MASK_COLOR = (0.0,0.0,1.0) # In BGR format

    def distance_contour_to_point(self, contour, point):
        # Returns the distance between the barycenter of a contour and a point
        center_x = int(np.mean(contour.reshape(-1, 2)[:, 0]))  # (x, y) format
        center_y = int(np.mean(contour.reshape(-1, 2)[:, 1]))  # not (row, col)
        return np.linalg.norm([center_x-point[0], center_y-point[1]])
        
    def mask_center_shift(self, image):
        # Returns the distance between the center of the mask and the center of the image
        # Image MUST be bw, full of 0 or 1
        if image.shape[2] != 4:
            raise AssertionError('The image needs to have 4 channels')
        image = image[:, :, 3]  # Grab the mask
        yc, xc = (image.shape[0]//2, image.shape[1]//2)
        row_multiplier = np.linspace(1, image.shape[0], image.shape[0]).reshape(1, -1)
        col_multiplier = np.linspace(1, image.shape[1], image.shape[1]).reshape(-1, 1)
        #print(row_multiplier, row_multiplier.shape)
        #print(col_multiplier, col_multiplier.shape)
        #print(image.shape)
        number_entries = int(np.sum(image)/255)
        image = image/255  # Get back to 0 or 1
        average_row = int(np.sum(np.matmul(row_multiplier, image))/number_entries)
        average_col = int(np.sum(np.matmul(image, col_multiplier))/number_entries)
        distance = int(np.linalg.norm([average_row-yc, average_col-xc]))
        return distance
        
    def image_contour(self, image, sorting='None', min_area=0, max_distance=np.inf):
        # Return the largest contour in an image
        # If using a zoom_box, keep the coordinates of the points in the referential of the input image.

        contour_info = []
        (xc, yc) = (image.shape[1]//2, image.shape[0]//2)
        contours = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        contours = imutils.grab_contours(contours)  # These are in the (x, y) format
        
        for c in contours:
            #print(type(c))
            #print(c[:10])
            # contour values are a numpy array
            contour_info.append((
                c,
                cv2.isContourConvex(c),
                cv2.contourArea(c),
            ))
            #print(contour_info[-1][0][:10])
            #assert 1==0
        contour_info = [c for c in contour_info if c[2] > min_area]
        contour_info = [c for c in contour_info if self.distance_contour_to_point(c[0], (xc, yc)) < max_distance]
        if sorting == 'area':
            contour_info = sorted(contour_info, key=lambda c: c[2], reverse=True)
        # Sorted by distance to center
        elif sorting == 'distance':
            contour_info = sorted(contour_info, key=lambda c: self.distance_contour_to_point(c[0], (xc, yc)))
            
        return contour_info
        
        
    def failed_processing(self, img, message, verbose=False):
        # Automate the return in case of failure
        if verbose:
            print(message)
        return 0, np.zeros((img.shape[0], img.shape[1], 4), dtype=np.uint8)
        
        
    def extract_positive(self, img, verbose=False):
        # img is a 3 channels RGB array
        # img_returned is a 4 channel RGB array
        img_area = img.shape[0]*img.shape[1]
        min_area = 0.1*img_area
        max_area = img_area
        
        if img.shape[2] != 3:
            raise AssertionError('Input image needs to have 3 channels')
        """[Show original]
        plt.figure()
        plt.imshow(img)
        plt.title("Original image")
        #plt.show()
        """
        
        #== Processing =======================================================================
        
        #-- Read image -----------------------------------------------------------------------
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)  # The input image is loaded with plt (RGB)

        #-- Edge detection -------------------------------------------------------------------
        # Returns a list containing info about the largest area contour
        edges = cv2.Canny(gray, self.CANNY_THRESH_1, self.CANNY_THRESH_2)
        #edges = cv2.dilate(edges, None, iterations=self.MASK_DILATE_ITER)  # Clean up edges
        #edges = cv2.erode(edges, None, iterations=self.MASK_ERODE_ITER)
        """[Show after Canny]
        plt.figure()
        plt.imshow(edges, cmap='gray')
        plt.title("After Canny")
        #plt.show()
        """
        #assert 1==0

        sorted_contour = self.image_contour(edges, sorting='area', min_area=25)
        #print([v[2] for v in sorted_contour])
        if len(sorted_contour):
            max_contour = sorted_contour[0]
        else:
            return self.failed_processing(img, "[WARNING] No contour found after running the Canny.", verbose=verbose)
        
        # Filter out the contours that are bigger than the original crop
        if max_contour[2] > max_area:
            message = "[WARNING] Max contour area {} which is larger than limit {}".format(int(max_contour[2]), max_area)
            return self.failed_processing(img, message, verbose=verbose)
        else:
            #-- Create empty mask, draw filled polygon on it corresponding to largest contour ----
            # I. Clean up the image with a fillPoly
            mask = np.zeros(edges.shape)
            image = cv2.fillPoly(mask, max_contour[0], 255)  # Outputs a float64 array
            h, w = image.shape[:2]
            bw_mask = np.array(image, dtype=np.uint8)
            
            # II. Copy that image and gradually fill it
            #bw_mask = image.copy()
            #bw_mask = image
            contour_info = self.image_contour(bw_mask, sorting='area')
            
            
            relevant_contours = [contour for contour in contour_info if contour[2] > min_area]
            #print("Contours lengths:", [len(contour[0]) for contour in relevant_contours])
            #print("Contours area:", [contour[2] for contour in relevant_contours])
            for contour in relevant_contours:  # Iterate in descending order
                #print("New contour of length {} and area {}".format(len(contour[0]), contour[2]))
                # 1. Find a point inside the contour to perform a floodFill
                contour_points = np.array(contour[0]).reshape(-1, 2)
                pixel_shift = np.array([[0, 1], [-1, 1], [-1, 0], [-1, -1], [0, -1], [1, -1], [1, 0], [1, 1]])
                seed_floodFill = (0, 0)
                for shift in pixel_shift:
                    potential_seed = tuple(contour_points[len(contour[0])//2] + shift)
                    # Not checking if the potential seed is still inside the crop
                    # Crop should be centered so 1 px difference should not matter.
                    #print(cv2.pointPolygonTest(contour[0], potential_seed, False))
                    if cv2.pointPolygonTest(contour[0], potential_seed, False) == 1:
                        # +1 is inside, -1 outside and 0 on vertex
                        seed_floodFill = potential_seed
                        break
                # 2. Try to fill the contour
                if seed_floodFill != (0, 0):
                    #print(seed_floodFill)
                    # Fill the inside of the image
                    #mask = np.zeros((h, w), np.uint8) 
                    window = np.zeros((h+2, w+2), np.uint8) 
                    _, bw_mask, _, _ = cv2.floodFill(bw_mask, window, seed_floodFill, 255)
                    """[Display gradual fill]
                    plt.figure()
                    plt.imshow(bw_mask, cmap='gray')
                    plt.scatter(contour[0].reshape(-1, 2)[:, 0], contour[0].reshape(-1, 2)[:, 1], s=1)
                    plt.title(bw_mask.shape)
                    plt.show()
                    """
                else:
                    return self.failed_processing(img, "[WARNING] Could not find a seed to fill the contour of this mask", verbose=verbose)
            
            
            #-- Additional post processing to improve the mask quality--------------------------------------------------------
            # These param were found to be optimal for my application
            #bw_mask = cv2.dilate(bw_mask, None, iterations=self.MASK_DILATE_ITER)
            #bw_mask = cv2.erode(bw_mask, None, iterations=self.MASK_ERODE_ITER)
            #bw_mask = cv2.GaussianBlur(bw_mask, (self.BLUR, self.BLUR), 0)
            
            # Additional erosion to clean up the edges and improve blending
            bw_mask = cv2.erode(bw_mask, None, iterations=1)
            
            """[Display final result]
            plt.figure()
            plt.imshow(bw_mask, cmap='gray')
            plt.title("Final image")
            plt.show()
            """
            #assert 1==0
            
            #print(mask.dtype, np.max(mask), np.min(mask))

            img         = img.astype('float32') / 255.0                 #  for easy blending
            # split image into channels
            c_red, c_green, c_blue = cv2.split(img)

            # merge with mask got on one of a previous steps
            img_a = cv2.merge((c_red, c_green, c_blue, bw_mask.astype('float32') / 255.0))
            
            # Convert back to RGB
            img_a *= 255
            img_a = img_a.astype(np.uint8)

            #img_a = np.array(img_a*255.0, dtype=np.uint8)
            # print("Extracted saved with range from {} to {}".format(np.min(img_a), np.max(img_a)))
            area_returned = int(np.sum(bw_mask)/255)  # Exact value of the area this time
            img_returned = img_a
            
            # SANITY CHECKS
            # Check for almost white image due to open contour
            if area_returned > 0.9*img.shape[0]*img.shape[1]:
                return self.failed_processing(img, "[WARNING] The whole image was filled in white during floodFill", verbose=verbose)
            elif area_returned < min_area:
                return self.failed_processing(img, "[WARNING] Area returned is below the {} threshold.".format(min_area), verbose=verbose)
            max_dist = int(min(img.shape[0], img.shape[1])//4)
            if self.mask_center_shift(img_returned) > max_dist:
                return self.failed_processing(img, "[WARNING] Resulting mask is off-center", verbose=verbose)

        """[Show result]
        plt.figure()
        plt.imshow(img_returned)
        plt.title("Final image")
        plt.show()
        """
        
        return area_returned, img_returned


    def blend_with_negative(self, negative, positive, rotations=(0, 0), scaling=(1, 0)):
        # param are understood as (mean, std_dev)
        # Rotation is taken from N(mean, std_dev)
        # Scaling is taken from abs(N(0, std_dev))+mean (only zoom_in)
        ps = positive.shape
        ns = negative.shape
        if ps[2] != 4:
            raise AssertionError('The positive image needs to have 4 channels')
        if ns[2] != 3:
            raise AssertionError('The negative image needs to have 3 channels')
        # positive image has 4 channels RGB, negative only 3 RGB
        if np.max(positive) <= 1:
            raise AssertionError('The positive image needs to be RGBA uint8')
        if np.max(negative) <= 1:
            raise AssertionError('The negative image needs to be RGBA uint8')
        # Returns an RGB array
        
        
        # 0. Resize the positive image to be compatible if needed
        if not np.array_equal(ps[:2], ns[:2]):
            # If crop smaller, just pad
            if ps[0] < ns[0] and ps[1] < ns[1]:
                # Equal padding on each side & support odd numbers
                padding = (((ns[0]-ps[0])//2, ns[0]-ps[0]-(ns[0]-ps[0])//2), ((ns[1]-ps[1])//2, ns[1]-ps[1]-(ns[1]-ps[1])//2), (0, 0))
                # pad with 0
                positive = np.pad(positive, padding, 'constant', constant_values=0)
                ps = positive.shape
            else:
                positive.resize((ns[0], ns[1], 4))
        
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
            shift = abs(np.random.normal(0, scaling[1]))
            scaling[0] = 1 if scaling[0] < 1 else scaling[0]  # Limit zoom to [1; +inf]
            zoom = scaling[0] + shift
            row_c = ps[0]//2
            col_c = ps[1]//2
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
