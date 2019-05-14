for i in {1..8}
do
    nohup python -m cProfile -o profiling_results process_file_multithreaded.py &
done
wait
nohup python main.py
