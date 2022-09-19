**Introduction:**

In Festo's CP Lab, there is a pick and place module which instructs the user to assemble a device based on the instructions on the screen. The parts are located in blue boxes which have a small led connected to them. The led lights up green indicating which box to pick out of. If the module detects that the individual has placed their hand on the wrong box, then the led flashes red. However, the module has no way of detecting if the user has correctly assembled the parts. For this a machine learning algorithm can be used to check assembly of the device.

The system created to detect this is a raspberry pi 4 which is mounted to the top of the machine. The Pi is fitted with a camera to be used with a machine learning algorithm for detecting incorrect or correct assembly. This paper will go over how to set up an image recognition system onto a Raspberry PI and have it communicate with the PLC in the CP-Lab.

**Mounting the Raspberry Pi on the Pick and Place:**

To set up the raspberry pi on the pick and place module simply insert the sliding screw holder into the small gap between the frames. Then use bolts to screw in the raspberry pi into the area. Make sure the camera of the Pi is right above where the order trays arrive.

**Training the Machine Learning System:**

Before training the system it is important to gather images that the ML system will be trained on. To do this gather around 15-20 pictures of every image that the ML algorithm will see. This means that you take 15-20 pictures of just a black front cover. Then take another 15-20 pictures of a black front cover with a PCB. Another, 15-20 of the same combination with the top fuse, rear fuse and both fuses at the same time. Repeat this process for every colour front cover. Since the back cover hides everything underneath it, it doesn't matter what is underneath it. Thus take 15-20 pictures of each colour of back cover present.

To take these pictures we will use a package for picamera. Since picamera doesn't come with a graphical interface, it is recommended to use github user Billwilliams1952's picamera GUI so that the pictures may be taken rapidly. The link to the module is here:

[https://github.com/waveform80/picamera](https://github.com/waveform80/picamera)

The resolution of the pictures are irrelevant because they can be resized based on the ML algorithm that is chosen in the future. For a standard definition choose 1280x720 but any resolution higher is fine too.

After you gather all the images the next step is to label them. To do this we will use labelimg which will be used to draw squares around each of the items. We will also need a text file with all the names of the classes present.

Classes:

- Empty
- Black Back Cover
- Red Back Cover
- Blue Back Cover
- Grey Back Cover
- Black Front Cover
- Red Front Cover
- Blue Front Cover
- Grey Front Cover
- PCB
- Top Fuse
- Rear Fuse

Here are the classes that should be used:

The link to the labelimg is here:

[https://github.com/tzutalin/labelImg](https://github.com/tzutalin/labelImg)

Here is an example of what the labeling in labelimg looks like:

![](RackMultipart20220919-1-i21vgx_html_d4e83926879ffc91.png)

**Training the Machine Learning System:**

There are multiple ways to train the machine learning algorithm. For this specific project we will use an already setup notebook by github user Nkap23. The link to the notebook is here:

[https://github.com/Nkap23/TensorFlow\_with\_Colab\_tutorial](https://github.com/Nkap23/TensorFlow_with_Colab_tutorial)

Choose a pretrained model from the tensorflow model zoo that will be used to train our model on top of. The reason we use a pretrained model is that it makes it much easier to train custom images on a pretrained model rather than creating a fresh new algorithm.

[https://github.com/tensorflow/models/blob/master/research/object\_detection/g3doc/tf2\_detection\_zoo.md](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md)

Due to the raspberry pi's low processing power, we need to choose a model that has a low speed. A lower speed indicates a faster system due to lower processing required. Luckily these frameworks are easy to identify due to them having mobile in their name such as "CenterNet MobileNetV2 FPN 512x512". For this guide we will use "SSD MobileNet v2 320x320".

Follow the steps on this medium article on how to set up the file structure to get ready to train the ML system. Stop on step 13 to make the edits below.

[https://medium.com/swlh/tensorflow-2-object-detection-api-with-google-colab-b2af171e81cc](https://medium.com/swlh/tensorflow-2-object-detection-api-with-google-colab-b2af171e81cc)

Configuring the pipeline:

On step 13 we want num\_classes to equal the number of classes we wrote in the labelImg in this case it would be 12. For batch\_size set it to a value around 64-256. For num\_steps set it to a value between 5,000 - 10,000 this is due to the fact that the simplicity of this system means that it doesn't need a large amount of steps for a large training time. If your system is more complex you can use anywhere between 50,000 - 200,000 steps.

For further reading batch\_size and num\_steps read this:

[https://stats.stackexchange.com/questions/164876/what-is-the-trade-off-between-batch-size-and-number-of-iterations-to-train-a-neu](https://stats.stackexchange.com/questions/164876/what-is-the-trade-off-between-batch-size-and-number-of-iterations-to-train-a-neu)

Then simply follow the instructions on the google collaborate file to train the system.

After training the system, download the folder named my\_model that is found under Tensorflow/workspace/training demo/exported-models. This saved model is the model that was trained in the previous step. Save it for now as it will be used when implementing the system on the Pi.

**Setting up the PLC:**

Before we create any function block we need to know some information about the PLC. Right Click the PLC and click under properties.

![](RackMultipart20220919-1-i21vgx_html_ae71862963ab40c9.png)

Under properties go under ethernet addresses and note down the IP address and subnet mask of the PLC. Additionally make sure to uncheck the User router block.

![](RackMultipart20220919-1-i21vgx_html_883c7ebe551d7fc3.png)

Next we are going to set up the function block which will communicate with the PI.

**Setting up the function block:**

To set up the system in the PLC we will construct a function block using TCON, TSEND and TRVC. These function blocks will be used to connect, send and receive information to the raspberry pi. To setup this create a function block and set it up as shown below. ![](RackMultipart20220919-1-i21vgx_html_26b7e5df23ae04d1.png)

After TCON has been set up, click the blue box icon, go to connections with these IP addresses. When setting up the TCON block the IP address on the right is the address of the raspberry pi. The IP address on the left is the IP address of the PLC. The partner port is the port that will be used to access the pi. The port number realistically can be anything. For further reading here is a link to information about ports:

[https://whatismyipaddress.com/port](https://whatismyipaddress.com/port)

The connection ID also can be any number as long as the number is consistent with the TSEND, TDISCON and TRCV block.

![](RackMultipart20220919-1-i21vgx_html_cbd62259512cd847.png)

Before we set up TSEND, TSDISCON and TRCV we have to create a data block that will store the information that is being sent from the PI. Create a data block with a variable data which is an array of characters from 0 to 255. The next variable is just a simple Hello World string that can be used for debugging purposes and can be omitted entirely.

![](RackMultipart20220919-1-i21vgx_html_1ca4ee910688b460.png)

Then right click the datablock and uncheck Optimized block access

![](RackMultipart20220919-1-i21vgx_html_daac659026f50f51.png)

Next up is to set up the TSEND, TDISCON and TRCV

![](RackMultipart20220919-1-i21vgx_html_bf9d2ddfea0c2cee.png)

![](RackMultipart20220919-1-i21vgx_html_6ad892ba21785464.png)

![](RackMultipart20220919-1-i21vgx_html_98ff6a2142f0eb72.png)

As you can see the ID for these blocks are the same as the one in the TCON block.

![](RackMultipart20220919-1-i21vgx_html_a5b019f6cf2b1c0a.png)

![](RackMultipart20220919-1-i21vgx_html_69096fba398f2339.png)

Network 6 sets the data where the Pi will be writing the T or F char to an empty string.

Network 7 sets a boolean called data\_bool to true if the Pi sends a T character.

Network 8 sets a boolean called data\_bool to false if the Pi sends a F character.

Network 9 sets a boolean called disp\_popup to true if the xorderconf is true and data bool is false. This means that there is an error and a display needs to popup. Otherwise the opposite is true.

Place this function block at the bottom of Main so that it is constantly running.

![](RackMultipart20220919-1-i21vgx_html_59f4f688c7dea6c0.png)

Next we need to write a script for the HMI so it displays a loading screen and an error message for when the PI detects an error. First go to the HMI and create a loading screen like this one.

![](RackMultipart20220919-1-i21vgx_html_478522aff225f944.png)

Then write the script as so. This script creates a loading screen for about 15 seconds so that the Pi has adequate time to check the workpiece.The if statement at the bottom checks if the error screen needs to be shown or not based on the boolean value of picheck. If there is an error orderconf is reset and the error popup is shown. If not then the carrier is sent on its way. There is a 2 second delay present in sending the carrier because a buffer time is needed to send the carrier. To send the carrier we need to set the xOrdConfirm bit to true then false.

![](RackMultipart20220919-1-i21vgx_html_c99144e86548c47a.png)

Next we need to configure the HMI screens so that an error message pops up. The following is the series of logic we will set up so that the error messages and loading screens will popup.

Here we set xorderconf2 to true when the confirm when the button is clicked so we know the order is confirmed. This signals the function block we wrote above so that the PLC knows if we need to show an error popup or send the carrier on its way.

![](RackMultipart20220919-1-i21vgx_html_1d5ef930ef1fa3e5.png)

We are also gonna create a SetBit on release if the first click is too fast for the HMI to recognize. We are also going to call the loading function that we created in VBScript earlier.

![](RackMultipart20220919-1-i21vgx_html_7b524a7774dd91dc.png)

Next we are going to create an error detected screen like so. Then under the click section of the ok button we are going to reset the disp\_popup. This will tell the PLC that it is no longer necessary to display the popup.

![](RackMultipart20220919-1-i21vgx_html_3e75448b29751611.png)

We are adding this same functionality under press under the situation that the click is not executed.

![](RackMultipart20220919-1-i21vgx_html_7e41b111dca70cb4.png)

Under release we will show the Popup ManWorkMes\_Popup. This will display the instructional screen so that the user can correct their error.

![](RackMultipart20220919-1-i21vgx_html_61abc6ae58ddfdd9.png)

Under override we will add a reset bit for xorderconf2 so we reset the system for the next order since this one is being overridden. We will also reset the disp\_popup so that the PLC knows that showing the error screen is no longer needed.

![](RackMultipart20220919-1-i21vgx_html_5c9d595b0557285c.png)

Next we will create a SetBit for xOrderConfirm.

![](RackMultipart20220919-1-i21vgx_html_efa3adef04cd0950.png)

Then we will create a ResetBit on xOrdConfrim. We will make this current popup screen disappear by turning the displaymode of the picheck\_good popup to Off. Then we will reset the disp\_popbit to false. This will allow the carrier to move forward and get rid of the current error popup.

![](RackMultipart20220919-1-i21vgx_html_be4f1ba49412266c.png)

**Setting up the Raspberry PI:**

First connect a raspberry PI to the PLC via an ethernet cable. This ethernet cable can be via a switch that routes to the PLC too.

To start setting up the raspberry pi first start a new terminal and type:

_ifconfig_

This is what it should look like

![](RackMultipart20220919-1-i21vgx_html_a7deefb982711194.png)

Here the eth0 IP address of the PI (inet) is 172.21.0.100 and the netmask is 255.255.192.0. If you recall the numbers in the TCON block, the IP address matches the value that was imputed into the TCON block. The netmask matches the submask net that was present in the properties section of the PLC.

If the values here don't match with the PLC you will need to change it by following these steps:

Open a new terminal and type:

_sudo nano /etc/dhcpcd.conf_

It should open up a file that looks like this:

![](RackMultipart20220919-1-i21vgx_html_9e6a6bc0fadd4ec6.png)

Scroll down using the arrow keys then edit the static ip address. The first few numbers should be the IP address that you wish to set the raspberry pi to have. Notice that the first two values of the IP (172.21) are the same as the first two values of the IP of the PLC.

This is important for the PLC to connect to the PI. The last integer after the backslash is the netmask of the PLC. We know our IP address is 172.21.0.100 from the PLC but we need the integer value for our subnet mask of 255.255.192.0 . For this we will use a subnet mask calculator

[https://www.calculator.net/ip-subnet-calculator.html](https://www.calculator.net/ip-subnet-calculator.html)

Within the IP Subnet Calculator

You should see two input boxes. The first is the subnet box, from clicking on this drop down choose the subnet that you would like to use. In our case it is 255.255.192.0. Then we input the IP address we need which is 172.21.0.100 .

![](RackMultipart20220919-1-i21vgx_html_2f62226b10c847.png)

Click Calculate and you see this

![](RackMultipart20220919-1-i21vgx_html_4f663bf92d4b4d75.png)

This value of short is what we will use for the static IP address for the Pi.

![](RackMultipart20220919-1-i21vgx_html_5be18bd004dcb363.png)

This is what the end file should look like. Simply click CTRL + S then CTRL + X to save and exit the file.

**Implementation on Raspberry PI:**

Download this zipfile: [https://drive.google.com/file/d/1xmldPTPA6PegB5fkOryFAjrAaaYoRRFA/view?usp=sharing](https://drive.google.com/file/d/1xmldPTPA6PegB5fkOryFAjrAaaYoRRFA/view?usp=sharing)

Unpack it under /home/ in the Pi

To implement the system into raspberry pi we will use an edited version of the camera module that was designed by github user armaanpriyadarshan.

This is the file named TF\_PiCamera\_OD.py. Open it and go down to line 27. As you can see the address is the same as the one of the raspberry pi that we set previously. The port is also the same as the one that was set in the PLC. Essentially, this is the address of the server that the PLC will connect to; so that the PLC and Pi can communicate with each other. If you have a different IP address and port to the one in this guide, make sure to change these values.

![](RackMultipart20220919-1-i21vgx_html_9241a7f77706e500.png)

Navigate to the tensorflow file and navigate to the od-models folder.

![](RackMultipart20220919-1-i21vgx_html_297fba355acc08d8.png)

Now paste the contents of my\_model into od-models. This should include the saved model, checkpoint and labelmap.pbtxt. Make sure the names of the folder match the ones shown below. The pipeline.config file is optional and not necessary for the code to work. It is simply used for reference.

![](RackMultipart20220919-1-i21vgx_html_c8b7c7910b763f43.png)

**Starting the System:**

Now it's time to start the system. Open a new terminal and type the following code:

_cd tensorflow_

_export PYTHONPATH=$PYTHONPATH:/home/pi/tensorflow/models/research:/home/pi/tensorflow/models/research/slim_

_source bin/activate_

_python TF\_PiCamera\_OD.py_

(Alternatively the code is also written in a text file called startup\_code.txt which is found in the tensorflow folder. Simply copy this and paste it into the terminal.)

This will start the system. As you can see the connection from 172.21.4.1 is the connection from the pick and place PLC. This IP address matches the IP address that was shown in the PLC when the TCON block was being set up.

![](RackMultipart20220919-1-i21vgx_html_bcbef1b06ed75ca4.png)

After about 3-4 minutes (based on the complexity of your ML program) the code should start and this should be the end result on your screen:

![](RackMultipart20220919-1-i21vgx_html_fd6f477087c26e6d.png)
