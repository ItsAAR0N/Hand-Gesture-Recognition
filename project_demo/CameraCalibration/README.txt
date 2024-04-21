Date:2021/05/27
这是利用opencv自带函数库进行摄像头标定的程序，步骤如下

1 首先需要准备标定板，可以运行GenerateCalibrationPlate.py生成需要的标定板
默认生成8x6的标定板，如果需要修改参数，可以修改CalibrationConfig.py文件

2 生成标定板后，需要用打印机打印出来，打印时需要选择1:1打印，打印之后将标定板
贴到平坦的物体上，之后需要移动它来进行采集图片

3 运行CollectCalibrationPicture.py来采集图片，移动标定板尽量让它出现在图像上的
各个位置，避免太边缘的位置，图片数量20-30张左右即可，如果效果不好再考虑增加图片数量

4 运行Calibration.py程序来进行标定，标定程序会筛选出不合格的图像，并且删掉，合格
的图像20-30张为佳，如果不够，可以继续采集补上，再重新标定

5 运行TestCalibration.py来测试标定的效果，标定后的图像显示的视野会比原来的小，如果要
在程序中调用，可以参考这个程序


Date: 2021/05/27
This is a program for camera calibration using the built-in OpenCV functions. The steps are as follows:

First, you need to prepare a calibration board. You can run GenerateCalibrationPlate.py to generate the required calibration board. By default, an 8x6 calibration board is generated. If you need to change parameters, you can modify the CalibrationConfig.py file.

After generating the calibration board, you need to print it out using a printer. When printing, you should choose 1:1 scale printing. After printing, attach the calibration board to a flat object, then you will need to move it around to collect images.

Run CollectCalibrationPicture.py to collect images. Move the calibration board to ensure it appears at various positions in the image, avoiding the too-peripheral positions. A total of about 20-30 images is sufficient. If the results are not good, consider increasing the number of images.

Run the Calibration.py program to perform calibration. The calibration program will filter out and delete any images that are not qualified. Having 20-30 qualified images is ideal. If there aren't enough, you can continue to collect more and then recalibrate.

Run TestCalibration.py to test the effect of the calibration. The field of view displayed by the calibrated images will be smaller than the original. If you want to call it in a program, you can refer to this program.