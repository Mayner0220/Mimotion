# Mimotion

## Intorduction
이 프로젝트의 목표는 사람의 감정 중 7개의 감정을 분류하는 것입니다. 모델에 사용되는 레이어는 DCNN(Deep Convolution Neural Network)이며, 데이터셋은 Kaggle에 오픈 데이터소스로 공개되어 있는 FER-2013를 사용했습니다.

(데이터셋에 대한 자세한 정보는 하단에 포함되어 있습니다.)

또한 이 프로젝트의 일부 내용은 atulapra의 Emotion-detection 프로젝트를 참조하여 개발되었습니다.

- Reference: [atulapra : Emotion-detection](https://github.com/atulapra/Emotion-detection)

## Skillset
- Tensorflow core v2.3.0
- Keras 2.4.0
- OpenCV 4.4.0.42
- Etc...

## Dataset
- Name: FER-2013
- License: Open Database
- Source: https://www.kaggle.com/msambare/fer2013
- Published: ICML(International Conference on Machine Learning)
- Consist: 35887 face images
- Size: 48x48 Size
- Categories: angry, disgusted, fear, happy, neutral, sad, surprised (7 categories)
