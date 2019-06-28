# Face-comparision-and-extraction
Pre-requisite
	* clone MTCNN github repo "git clone https://github.com/ipazc/mtcnn.git"
	* pip3 install mtcnn
	* pip3 install tensorflow opencv-contrib-python
  * Pip3 install keras_vggface
  * Pip3 install face_recognition
  * sudo apt-get install build-essential cmake
  * sudo apt-get install libgtk-3-dev
  * sudo apt-get install libboost-all-dev

1. Aadhar Face Image Extraction
	* Logic/ tools used-
		* Used pre-trained "MTCNN-FACE-DETECTION-MODEL"
		* Applied face-detection to detect face coordinates
		* Then sliced the face image
	* working
		* Update image-path in the code
		* Run "python3 aadhar_face.py"


2. Face Comparision
	* Logic/ tools used-
		* Each face have their own unique face features like distance between eyes, nose position etc.
		* Face can be compared/matched based on their features
		* Used pre-trained "MTCNN-MODEL" to generate face feature matrix
		* Takes two images as an input, generates "face feature matrix" of both the images by passing it through "neural network fully connected layers"
		* The fully connected layer model gives feature matrix as an output 
		* Similarity between two images can be calculated based on calculating eucledean distance between their face feature matrix i.e. more the distance lesser the similarity
		* Used average value of "Cosine-Distance-Fucntion" and "Eucledean-Distance" to calculte the distance between both the matrices 
		* Cosine-distance function keeps the distance between (0,1) range increasing efficiency and ease of setting threshold(Currently set to 0.6) limit
		* Lower the distance more similar faces are.
	* Working 
		* Update input images paths in the code
		* Run "python3 face_verification.py"
