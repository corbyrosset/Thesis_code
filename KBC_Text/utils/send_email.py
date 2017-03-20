import yagmail

def send_notification(subject, filename):
	'''
		send an email containing the log file. Filename is a path. 
	'''
	yag = yagmail.SMTP('corbylouisrosset@gmail.com', )
	yag.send('corbylouisrosset@gmail.com', subject = subject, contents = filename)

