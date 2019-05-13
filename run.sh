for i in {1..8}
do
    nohup python process_file_multithreaded.py &
done
wait
nohup python main.py
