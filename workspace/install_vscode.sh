#1 Vscode 설치
wget "https://update.code.visualstudio.com/latest/linux-deb-arm64/stable" -O code_arm64.deb
chmod +x code_arm64.deb
sudo dpkg -i code_arm64.deb
sudo apt --fix-broken install

