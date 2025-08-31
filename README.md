# Fake Guard - AI Content Detection System 🛡️

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28.0-red)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8.0-green)
![Transformers](https://img.shields.io/badge/🤗%20Transformers-4.35.0-yellow)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

A powerful web application that detects AI-generated content in text and videos using advanced machine learning models.



</div>

## ✨ Features

### 📝 Text Detection
- **AI-Generated Text Identification**: Detects content from GPT-3, GPT-4, and other LLMs
- **Confidence Scoring**: Provides percentage-based AI likelihood scores
- **Real-time Analysis**: Instant results with visual progress indicators
- **Smart Thresholding**: Intelligent classification with multiple confidence levels

### 🎥 Video Detection  
- **Deepfake Detection**: Identifies AI-manipulated videos and synthetic media
- **Multi-Frame Analysis**: Processes up to 100 frames for comprehensive detection
- **Face Recognition**: Advanced facial feature analysis using OpenCV
- **Frame-by-Frame Results**: Detailed breakdown of suspicious frames

### 🎨 User Experience
- **Clean Modern UI**: Streamlit-based responsive interface
- **Intuitive Navigation**: Simple two-mode selection (Text/Video)
- **Visual Feedback**: Progress bars, metric cards, and color-coded results
- **Mobile Friendly**: Responsive design works on all devices

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/your-username/fake-guard.git
cd fake-guard
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Run the application**
```bash
streamlit run app.py
```

4. **Open your browser** and navigate to `http://localhost:8501`

## 📋 Requirements

The application requires the following Python packages:

```txt
streamlit==1.28.0
transformers==4.35.0
torch==2.1.0
opencv-python==4.8.0
pillow==10.0.0
numpy==1.24.0
```

## 🛠️ Usage

### Text Detection
1. Select "Text Detection" from the sidebar
2. Paste your text into the input area (minimum 10-15 words recommended)
3. Click "Analyze Text" 
4. View the AI likelihood score and detailed verdict

### Video Detection
1. Select "Video Detection" from the sidebar  
2. Upload a video file (MP4, MOV, AVI formats supported)
3. Click "Analyze Video"
4. Monitor the progress as frames are processed
5. Review the results showing detected faces and suspicious frames

## 🏗️ Architecture

```
fake-guard/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── README.md             # Project documentation
└── .gitignore           # Git ignore rules
```

### Model Architecture
- **Text Detection**: Microsoft DialogRPT-updown model
- **Video Detection**: Custom CNN-based deepfake detection model
- **Face Detection**: OpenCV Haar Cascades
- **Framework**: Hugging Face Transformers + PyTorch

## 🔧 Technical Details

### Text Detection Pipeline
1. Input text preprocessing and tokenization
2. Model inference using DialogRPT
3. Score calculation and normalization  
4. Confidence-based classification
5. Visual result presentation

### Video Detection Pipeline
1. Video file upload and temporary storage
2. Frame extraction and face detection
3. Facial region preprocessing
4. Deepfake model inference per frame
5. Aggregation and result compilation

## 📊 Performance

- **Text Analysis**: ~2-5 seconds for typical paragraphs
- **Video Analysis**: ~1-3 minutes for 100 frames
- **Accuracy**: High detection rates for common AI models
- **Scalability**: Efficient memory usage and processing

## 🌟 Key Benefits

- **Real-time Detection**: Instant analysis without lengthy processing
- **High Accuracy**: Advanced models trained on diverse datasets  
- **User-Friendly**: Intuitive interface requiring no technical expertise
- **Privacy Focused**: Local processing - your data never leaves your machine
- **Open Source**: Transparent methodology and community-driven improvements

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 🐛 Known Issues

- Large video files may require more processing time
- Very short text samples may yield less accurate results
- Some advanced AI models might evade detection

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Hugging Face** for transformer models and pipeline
- **OpenCV** for computer vision capabilities
- **Streamlit** for the amazing web framework
- **Microsoft** for the DialogRPT model

---

<div align="center">

**Made with ❤️ using Python and Streamlit**

⭐ **Star this repo if you find it useful!**

</div>
