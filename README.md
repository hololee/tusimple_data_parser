Preprocessing for Tusimple_Lane_Detection data set
---------------------------------------------------
<`http://benchmark.tusimple.ai/#/`>

This python script is preprocessing code for tusimple lane detection data set.  
You can download the data in website and check the data set zip file.  

I used some code in this stackoverflow page for drawing line.  
<https://stackoverflow.com/questions/31638651/how-can-i-draw-lines-into-numpy-arrays?rq=1>

Need the following packages.

>1. numpy
>2. cv2
>3. scipy
>4. matplotlib (option for drawing image)

* * *

You can use this script like this.
```
import tusimple_data_parser as tdp

tdp.pre_processing(data_location='./label_data_0313.json',
                   processed_images_location='./image/image{0:04d}.png',
                   processed_labels_location='./label/label{0:04d}.png',
                   line_weights=10,
                   plot_images=False)
```
