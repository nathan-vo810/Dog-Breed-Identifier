import classifier
import server

def start_all_processes():
	print("Starting server...")
	server.start_process()

	print("Staring classifier...")
	clf_process = classifier.start_process()

def main():
	start_all_processes()

if __name__ == '__main__':
	main()