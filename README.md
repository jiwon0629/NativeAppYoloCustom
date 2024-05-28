# nativeAPP This is a yolov5's Gui program
1. 새폴더 만들고 CMD 에서 (또는 vscode 터미널에서) 소스코드를 복사한다.
```
git clone 해당 레포 주소
```
2. 새로운 개발환경을 만든다.
```
conda create -n jiwon python=3.9
```  
3. 콘다 Activate 또는 vscode 에서 개발환경 연결해준다.
```
conda activate jiwon
```
4. 개발환경 Package 설치
```
pip install -r requirements.txt
```
5. yolov5를 클로닝해온다.(REPO에서는 보이지 않지만)
```
git clone https://github.com/ultralytics/yolov5.git
```
## 폴더구조는
![codeResult](https://github.com/jiwon0629/NativeAppYoloCustom/assets/149983498/8656690a-41e9-4c60-b0b0-7251d160d9d6)

실행은 windowAPP.py
## 프로그램 실행 결과 화면
![person](https://github.com/jiwon0629/NativeAppYoloCustom/assets/149983498/c0ade085-4e57-4c87-9611-879f6d419836)
![CellPhone](https://github.com/jiwon0629/NativeAppYoloCustom/assets/149983498/7e5dce57-3e28-43a0-911c-0660c1d9cb6b)
https://github.com/jiwon0629/NativeAppYoloCustom/assets/149983498/0d13c9a3-cc71-4c87-8efb-1a275ceb2c41
https://github.com/jiwon0629/NativeAppYoloCustom/assets/149983498/169461dd-a2db-485e-852b-7d8a73d58fc7
위의 그림처럼 Person을 인식했을때 로봇의 19번 동작이 실행되고 CellPhone을 인식했을때 로봇의 17번 동작이 실행된다.



