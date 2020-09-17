# Face recognition in presence of mask

The task in this project is to match a face image without any occlusion with a face image covered with one of these masking approaches. This repo is a solution of summer challenge given by IITJ.

## Demo 

<p>
<p align="left"><img height="250" width=200" src="https://github.com/Garvit-32/Face-recognition-in-presence-of-Mask/blob/master/readmeImages/01-1.jpg" alt="01-1"/><img src="https://github.com/Garvit-32/Face-recognition-in-presence-of-Mask/blob/master/readmeImages/arrow.png" alt="arrow"/><img height="250" width=200" src="https://github.com/Garvit-32/Face-recognition-in-presence-of-Mask/blob/master/readmeImages/01-2.jpg" alt="01-2"/>
</p>
<p align="left"><img height="250" width=200" src="https://github.com/Garvit-32/Face-recognition-in-presence-of-Mask/blob/master/readmeImages/02-1.png" alt="02-1"/><img src="https://github.com/Garvit-32/Face-recognition-in-presence-of-Mask/blob/master/readmeImages/arrow.png" alt="arrow"/><img height="250" width=200" src="https://github.com/Garvit-32/Face-recognition-in-presence-of-Mask/blob/master/readmeImages/02-2.png" alt="02-2"/>
</p>
</p>


## Steps to run 
 1. Clone this repository
 ```
 git clone https://github.com/Garvit-32/Face-recognition-in-presence-of-Mask.git
 ``` 
 2. Install all the dependencies
```
pip install -r requirements.txt
```

 3. Download and paste the pretrained weight of face detection model in ./face_detection/weights/
using the [link](https://drive.google.com/file/d/1WeXlNYsM6dMP3xQQELI-4gxhwKUQxc3-/view)
 4. Download the hand made dataset using the [link](https://drive.google.com/drive/folders/1bhyhYWd4tqTmqEeJKzQAMv8q_Batyh0Z?usp=sharing)  ~ 110 MB
 5. Extract and paste it in project root directory
 6. (optional) You can use your dataset also but it follow a specific structure of directory
```
-dataset/
	-person_name_1/
		-image_01
		-image_02
	-person_name_2/
		-image_01
		-image_02
```
 7. Run generate_database python file to virtually append mask on person face in database
```
python generate_database.py
```
 8. Train the model
```
python train.py
```

## Steps to Test it out

 1. Extract face in image using extract_face.py 
 edit test_data_path = "to_your_test_path"
 ```
 python extract_face.py
 ```
 2. Test your image
 ```
 python test.py
 ```

