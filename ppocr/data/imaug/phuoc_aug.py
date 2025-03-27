
import albumentations as A

class RandomContrastBrighness(object):
    def __init__(
        self,
        p ,
        brightness_limit,
        contrast_limit,
        **kwargs,
    ):
        
        self.random_contrast = A.RandomBrightnessContrast(brightness_limit,contrast_limit=contrast_limit,p = p,**kwargs)
       
    def __call__(self, data):

        img = data["image"]
        img = self.random_contrast(image = img)['image']
        # Update data dictionary
        data["image"] = img
       
        
        return data
class MotionBlur(object):
    def __init__(
        self,
        p,
        blur_limit,
        **kwargs,
    ):
        self.motion_blur = A.MotionBlur(blur_limit=blur_limit, p=p,**kwargs)
        
    def __call__(self,data):
        img = data["image"]
        img = self.motion_blur(image = img)['image']
        # Update data dictionary
        data["image"] = img
        return data
    

class GaussianBlur(object):
    def __init__(
        self,
        p,
        blur_limit,
        **kwargs,):
        self.gaussian_blur = A.GaussianBlur(blur_limit=blur_limit, p=p,**kwargs)
    def __call__(self,data):
        img = data["image"]
        img = self.gaussian_blur(image = img)['image']
        # Update data dictionary
        data["image"] = img
        return data
    

class RandomBlur(object):
    def __init__(
        self,
        p,
        motion_blur_limit,
        gau_sigma_limit,
        gau_blur_limit,
        **kwargs,
    ):
        self.main =A.OneOf([
            A.GaussianBlur(sigma_limit=gau_sigma_limit,blur_limit=gau_blur_limit,p  = 1),
            A.MotionBlur(blur_limit= motion_blur_limit,p = 1),
        ],p = p)
    def __call__(self,data):
        img = data["image"]
        img = self.main(image = img)['image']
        # Update data dictionary
        data["image"] = img
        return data
    


class RandomColor(object):
    def __init__(
        self,
        p,
        saturation,
        hue,
        **kwargs
    ):
        self.main = A.ColorJitter(brightness=0,contrast=0,saturation=saturation,hue = hue, p = p)

    def __call__(self,data):
        img = data["image"]
        img = self.main(image = img)['image']
        # Update data dictionary
        data["image"] = img
        return data
    
    
class RandomNoise(object):
    def __init__(
        self,
        p,
        min_value,
        max_value,
        **kwargs
    ):
        self.main = A.AdditiveNoise(noise_type="gaussian",
								spatial_mode="per_pixel",
								noise_params={"mean_range": (0.0, 0.0), "std_range": (min_value,max_value)} ,
								p =p)

    def __call__(self,data):
        img = data["image"]
        img = self.main(image = img)['image']
        # Update data dictionary
        data["image"] = img
        return data
    

class RandomGammar(object):
    def __init__(
        self,
        p,
       gamma_limit,
        **kwargs
    ):
        self.main = A.RandomGamma(p= p ,gamma_limit=gamma_limit)

    def __call__(self,data):
        img = data["image"]
        img = self.main(image = img)['image']
        # Update data dictionary
        data["image"] = img
        return data