# Auto-Trimming Tennis Match

<p align="center">
  <img src="![alt text](786A9DD4-7960-46E6-A814-0F84889DFF90_1_105_c.jpeg)" alt="image" style="display:block; margin:auto;" />
</p>

 Automatically trims full tennis match recorind into highlight video as part of the developing tennis analytical website amied to make tennis more accessible by making self analysis easir. 




### Features 
-  Ball tracking using GridTrackNet
-  Players detection using a fine-tuned YOLOV8 model 
-  Interacting court key points label system 
-  Bounce auto-detections
-  Point in play detection
-  Extration of point in play segments 
-  Main pipline producing auto-trimming in one command 



## Future Improvements
-  Due to YOLOV8's slow nature, detecting players in each frame takes long time to run (which makes this program unpractical to run on vidoes longer than 5 min). Future improvement could improve model's efficieny by skipping frames or achiving an overall faster model performance.
-  Due to GridTrackNet's limitation to accurate detect the ball, there are errors in reading postions of the ball which also causes inacurracy in bounce and point detections. To improve accuracy of video trimming feature, implementions of algorithms that better interpolate the ball is needed.
- This program prodcues mulitple intermeidate files (such as posistion of ball/players/bouces) which uses lots of memory. 


## Steps
1. Check Python virsion
 GridTrackNet requires Python 3.9–3.11:
```bash
python3 --version
```

if your virsion is too old:
```bash
brew install python@3.11
```

2.  Create and activate virtual environment:
```bash
python3 -m venv tf-env
source tf-env/bin/activate
```
Upgrade pip inside virtue environment:
```bash
pip install --upgrade pip
```

3. Install required libaries:
```bash
pip install -r requirements.txt
```

4. Create input and output folders in directory:
```lua
input/
output/
```

5. Place your match video into input folder:
```lua
input/<your_vidoe>
```

6. Run with following command:
```bash
python main.py \
  --video_path "input/<your_video.mp4>" \
  --output_dir "output/" \
  --model_path "models/model_weights.h5" \
  --display_trail
```

7. When prompted, label court key points in following order(it will not work if order is wrong):
- Top left doubles corner 
- Top right doubles corner  
- Bottom left doubles corner
- Bottom right doubles corner 
- Top left singles corner 
- Top right single corner
- Bottom left singles corner 
- Bottom right singles corner 
- Top left service corner 
- Top right service corner 
- Bottom left service corner 
- Bottom right service conor 
- Top T line 
- Bottom T line 


## Output
Results (trimmed video + intermediate data) are saved in output/:
output/
├── <your_video>_bounces.json
├── <your_video>_court_keypoints.json
├── <your_video>_players.json
├── <your_video>_segments.json
└── <your_video>_shortened.mp4


*Disclaimers*:
- *Some parts of the code are developed with Chat-GPT's assistance and might include undifed behaviors.
- *Suitable to run on Mac with M2 chips but does not guarantee performance on any devices.
- This project is a learning experience and might (and very likey) contains errors.

## Acknowledgements
I would like to thank the following people for their contributions and support on this project:
Rebecca Kong – Collaborated on designing the PointView program and user interface.
Emily Ing – Collaborated on designing the PointView program and user interface.
Prof. Dodds – Provided guidance and mentorship throughout the project.
Peers in the research lab – Offered valuable feedback and inspiration for this project.
Harvey Mudd College – Provided tremendous opportunities and resources that made this project possible.

## References
- V. Korpelshoek, GridTrackNet, https://github.com/VKorpelshoek/GridTrackNet
- YOLOV8, https://yolov8.org
- yastrebksv, TennisProject, https://github.com/yastrebksv/TennisProject
- Code in a Jiffy, Build an AI/ML Tennis Analysis system with YOLO, PyTorch, and Key Point Extraction, https://www.youtube.com/watch?v=L23oIHZE14w



