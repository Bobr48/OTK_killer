import RPi.GPIO as GPIO
import time
GPI_up = 26
GPI_enc = 20
GPO_dpt = 21
GPO_red = 16
GPO_mark = 13
GPO_yellow = 19

count_test = 0
def get_rpm(c):
	global count_test
	if count_test==0:
		count_test = count_test + 1 # increase counter by 1
	else:
		count_test = count_test + 1

       
DELAY_TIME = 1
GPIO.setmode(GPIO.BCM)
GPIO.setup(GPI_up, GPIO.IN)
GPIO.setup(GPI_enc, GPIO.IN)
GPIO.setup(GPO_dpt, GPIO.OUT)
GPIO.setup(GPO_red, GPIO.OUT)
GPIO.setup(GPO_mark, GPIO.OUT)
GPIO.setup(GPO_yellow, GPIO.OUT)


GPIO.setmode(GPIO.BCM)       # Numbers GPIOs by physical location
GPIO.setup(GPI_enc, GPIO.IN)
GPIO.add_event_detect(GPI_enc, GPIO.FALLING, callback=get_rpm) # execute the get_rpm function when a HIGH >
 


def init():
	GPIO.output(GPO_dpt, GPIO.LOW)
	GPIO.output(GPO_red, GPIO.LOW)
	GPIO.output(GPO_yellow, GPIO.LOW)
	GPIO.output(GPO_mark, GPIO.LOW)
	
def up():
	return GPIO.input(GPI_up)
	
def enc():
	return GPIO.input(GPI_enc)
		
def dpt(x):
	if x == True:
		GPIO.output(GPO_dpt, GPIO.HIGH)
	if x == False:
		GPIO.output(GPO_dpt, GPIO.LOW)
	pass
	
def yellow(x):
	if x == True:
		GPIO.output(GPO_yellow, GPIO.HIGH)
	if x == False:
		GPIO.output(GPO_yellow, GPIO.LOW)
	pass

def red(x):
	if x == True:
		GPIO.output(GPO_red, GPIO.HIGH)
	if x == False:
		GPIO.output(GPO_red, GPIO.LOW)
def mark(x):
	if x == True:
		GPIO.output(GPO_mark, GPIO.HIGH)
	if x == False:
		GPIO.output(GPO_mark, GPIO.LOW)


