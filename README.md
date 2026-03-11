### **The Starting Point:** What was your prior experience with Computer Vision or Python before Friday?

I have some limited experience from college classes in the early 2010s based on computer vision, where we used now outdated models, and some more recent classification projects both on my own free time and related to workshops run at the Westminster office. During one of those workshops my team was able to use Azure ML to run a model which identified receptacles on electrical drawings in only 3 days of work, including hand-tagging training data. The advances over the almost decade between my initial foray into computer vision and more recent experiences have really impressed me. Not only are the results hugely improved with better accuracy and computation time, the accessibility and breadth of information available now makes for a much more welcoming environment. I have also done some recent work with OpenCV for the drawing compare functions used in LiveCount and Estimation MEP. OpenCV is the computer vision library we use for aligning drawings.

Whenever it has been the best fit for the problem I am trying to solve, whether because of the available libraries or the available resources, I have used Python on occasion throughout the years. I am by no means an expert in Python, but I am comfortable with the dynamically typed nature and the flexibility and power of the language when I work with Python. In fact, the ability of Python to kind of float between a strictly functional and more procedural mindset is one of my favorite things about the language, as you can hack together scripts very quickly while it also supports building maintainable and “safer” code. While the nature of Python can sometimes produce difficult to debug runtime errors, I am confident in my ability to debug and iterate on Python code. 

As a fun aside, the first code I ever wrote was in Python to create scripts for StarCraft: Brood War custom maps, and I often cite that as the seed of me ultimately becoming a software engineer\!

### **The Research Path:** List resources (articles, YouTube videos, documentation, StackOverflow threads) you used to learn.

[https://www.hitechbpo.com/blog/top-object-detection-models.php](https://www.hitechbpo.com/blog/top-object-detection-models.php)

[https://datature.io/blog/yolo26-the-edge-first-evolution-of-real-time-object-detection\#:\~:text=Our%20Developer's%20Roadmap-,Introduction%20to%20YOLO26,%2C%20and%20resource%2Dconstrained%20environments](https://datature.io/blog/yolo26-the-edge-first-evolution-of-real-time-object-detection#:~:text=Our%20Developer's%20Roadmap-,Introduction%20to%20YOLO26,%2C%20and%20resource%2Dconstrained%20environments).

[https://docs.ultralytics.com/compare/efficientdet-vs-yolo26/\#model-lineage-and-authorship](https://docs.ultralytics.com/compare/efficientdet-vs-yolo26/#model-lineage-and-authorship)

[https://www.geeksforgeeks.org/computer-vision/mmdetection-in-computer-vision/](https://www.geeksforgeeks.org/computer-vision/mmdetection-in-computer-vision/)

[https://www.youtube.com/watch?v=QzY57FaENXg](https://www.youtube.com/watch?v=QzY57FaENXg&t=53s)

[https://pub.aimind.so/beyond-yolo-implementing-d-fine-object-detection-for-superior-precision-a695523c26c7](https://pub.aimind.so/beyond-yolo-implementing-d-fine-object-detection-for-superior-precision-a695523c26c7)

[https://www.youtube.com/watch?v=YJP5XzEKS80](https://www.youtube.com/watch?v=YJP5XzEKS80)

[https://www.youtube.com/watch?v=0C6EnFq2XSY](https://www.youtube.com/watch?v=0C6EnFq2XSY) 

[https://github.com/roboflow/rf-detr](https://github.com/roboflow/rf-detr)

[https://keras.io/keras\_hub/](https://keras.io/keras_hub/)

[https://github.com/Peterande/D-FINE](https://github.com/Peterande/D-FINE) 

[https://huggingface.co/](https://huggingface.co/) 

### **The `Pivot" Moments:** Describe one specific technical hurdle you hit (e.g., "The model wouldn't load because of a version mismatch") and exactly how you searched for and found the fix.

The biggest hurdle I ran into was setup with the D-FINE model using Keras Hub which was described as an “easy” setup option as it provides pretrained models based on industry standard datasets like COCO. Since my machine is a Windows machine, and some of the required libraries, such as tensorflow, are dependent on being in a Linux environment, this caused a handful of problems which were not intuitive to resolve at first glance.

I ran into a number of issues with this Linux requirement trying to run the model successfully. This was something that was not really mentioned in the documentation for Keras Hub, so it was not even on my radar when I first started running into errors. The aforementioned tensorflow library was the first sign of an issue. I was able to download tensorflow using pip without issue but a dependency on tensorflow-text was not so easily resolved. A (too) brief initial Google search claimed there were ”easy” workarounds for Windows so I started trying those, changing the model parameters and configuration to avoid certain libraries or manually shimming in a Windows-friendly equivalent. After fighting error after error trying to get the code running natively on Windows, using workarounds that only slightly progressed the errors, I decided to take a step back and see if I could get a Linux environment on my machine and save myself the pain. I already had WSL2 installed and had some experience mimicking a Linux environment from debugging errors for some of our services hosted on Linux machines, so I turned my attention to that. Within a half hour, using the help of Gemini, I was able to get VSCode running straight out of my WSL virtual machine. Once I moved my code to that location and undid all the hacks I had implemented to get it working in Windows, I was able to run my code and get results out of the model. 

### **Model Choice:** Why did you choose the specific model you used?

Starting research around classification problems in computer vision pointed me repeatedly back to the YOLO model, perhaps based on the phrasing of queries I made to Gemini and search engines, and because it has emerged as a leader in the computer vision space over the last few years. However, when I dug more into that model, it seemed mostly optimized for speed so that the model could be run efficiently on each frame of a video or other real time data, and a bit dated. Another model which stuck out to me but I decided against using was RF-DETR. It is something I would like to look into more but based on time constraints, I ultimately chose not to use it as it did not seem as beginner friendly as the model I ultimately decided to use, **D-FINE**. RF-DETR and D-FINE have similar benchmarks in a lot of cases but RF-DETR shines for more domain specific applications with additional training and setup, whereas for this task of identifying dogs, cats, and people D-FINE often has slightly higher precision.

D-FINE models seemed like a good fit for this task especially because the “smaller” models are able to be run on a CPU rather than requiring tons of GPU computing power. This was important to me as I am running the model on my laptop and want to be able to iterate without high costs or downtime waiting for outputs. Additionally, my experience at Trimble and generally as a software engineer has been that we often have to balance cost versus output and usually we end up at a “happy medium” which this model fits into. D-FINE also has different versions out of the box and available using the Keras Hub library which use different levels of computational power, so allows for scalability without much configuration change. 

More generally, I like the idea of the transformer models over CNN both for this task and within the construction and estimation space as they claim to take the context of the image as a whole into consideration, while CNN (like YOLO or ResNet models) focus primarily on local features. A lot of upcoming work for Estimation seems at a surface level that it would benefit from that context, such as recognizing rooms/spaces and classifying entire drawings based on what they contain. Of course those items will require their own investigations and solutions, but I kept that context in mind while choosing my model for this challenge. 

There are many available models that provide different levels of output. Based on the specific requirements of simply identifying the existence (or lack thereof) of a dog, cat, or person, I decided early on against using some of the more intensive and complex models available. Some more configurable or powerful models I read about on hugging face seemed like overkill, especially considering the time sensitive nature of this challenge and the fact that some of them require powerful GPU computation. For other use cases, such as if we needed a tighter fitting bounding area or faster computation for work on video inputs, I would have likely chosen differently. 

### **If I had more time:** If you had more than 48 hours, what would you do next to complete this?

My solution works well for images either provided on the file system or through a URL, but has trouble with the live stream or webcam without doing some real troubleshooting or manually finding a link from an online stream, which is not always straightforward. In my case I had to do some setup to get my webcam cast to WSL that I did not write up as it is very specific to my situation, but for completion sake I would want to streamline that so it works more intuitively “out of the box”. 

Ideally, instead of running from a command line out of an environment that requires some setup, I would like my solution to be packaged up or hosted online so there is no setup required and a user can just put a link or upload a file. For the purposes of this challenge I was more interested in adding functionality and getting everything working, but it is not exactly user friendly if someone wanted to run it themselves easily.

Given more time I would do more extensive testing with the different options available through Keras Hub using this DFINE model. Currently I have it set to “small” but there are other options and I would try to see if the quality gets better with the larger models and also what the tradeoff is. I started this a little and I left my testing script in the repo just as an example, but I did not have enough time to really dig into it. Keras Hub also provides a ton of options for refining and training your own models, as it abstracts some of the options but also allows for changing out different aspects of the model for quick iterations.

Ultimately, if DFINE is the model to choose, I would also look into the benefits and trade off of downloading the code directly from the original D-FINE git repository to remove the reliance on the Keras Hub third party library. It certainly saved me time and effort for the sake of this challenge, but for a production ready solution I may want to train and host the model directly in order to have more control over the model and protect from any issues using Keras Hub. There is a ton of documentation and examples of this available via the git repo but it was out of scope for this challenge. 

I would also like to compare these results against the results of some of the other models available to see the difference in speed and accuracy. I had to decide with a time constraint on this model, but it may not be the best in practice. On that note, I would also ask some more questions on the actual use case for this to tailor my solution to the use case. As this is a basic challenge I know it is silly, but being able to understand the actual use case of any code has a significant impact on the final solution provided.

CoPilot helped with a lot of my code, and while I could explain what each line of code does there is still some general cleanup I would do with more time. I tried to simplify any overly verbose code and include comments wherever possible but I would not feel comfortable putting this into a shared code base without making sure I understood best practices for Python code more clearly and implemented them. 

### Setup steps:

**Running on Windows without WSL already set up (skip to next step if in Linux environment)**

WSL the easiest way to run in Linux environment on Windows machine by opening powershell as administrator and `wsl –-install`  
Once you have wsl, you can install Ubuntu with `wsl.exe –-install Ubuntu`  
Then, you can search “Ubuntu” through windows start menu and open an Ubuntu terminal

**Start here if you already have Linux environment**

Download the git repository (https://github.com/WillPeck-Trimble/AIML-Specialization-Challenge)  
Navigate to root folder  
Ensure your Python version is 3.12

**If you want to use venv to encapsulate your dependencies (optional)**

`sudo apt update` to update package list  
`sudo apt install python3.12-venv` to get venv for python 3.12  
`python3 \-m venv .venv` to initialize venv virtual environment  
`source .venv/bin/activate` to enter venv environment

**Install dependencies**

`pip install \-r requirements.txt`

**Now you can run the script to detect cats, dogs, and people in images\!**

I’ve included a couple of example images to run out of the box, or you can add more files, use URLs, or if you are feeling ambitious even take an image live from your webcam\!

**Examples:**

`python3 dfine_s_coco.py --file [filename]`
`python3 dfine_s_coco.py --url [url]`