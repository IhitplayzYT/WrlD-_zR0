pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

if [ $? -eq 0 ];then
    echo "Torch Install Finished"
else
    echo "Torch Install Failed! Manual Intervension Required"
fi

pip install -r requirements.txt

if [ $? -eq 0 ];then
    echo "Installed Requirements.txt Libs"
else
    echo "Failed to install Requirements.txt Libs"
fi
