@echo off
call "C:\Users\User\anaconda3\Scripts\activate.bat" "C:\Users\User\anaconda3"
call conda activate py311
python rename_extractors.py > log.txt 2>&1
echo Done >> log.txt
