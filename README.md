# easyViT:OpenCLIP과 DINOv2를 활용한 텍스트-이미지 매칭 및 벡터 검색

easyViT는 OpenCLIP과 DINOv2 모델을 활용하여 유사도 기반 검색 시스템을 구축하는 것을 목표로 합니다. 이 프로젝트는: <br/>
- [Reproducible scaling laws for contrastive language-image learning](https://arxiv.org/abs/2212.07143)<br/>
- [DINOv2: Learning Robust Visual Features without Supervision](https://arxiv.org/abs/2304.07193)<br/>

위 두 논문의 모델들의 구현을 포함하고 있습니다.

사용자가 텍스트를 입력하면 OpenCLIP을 이용해 텍스트 임베딩을 생성하고, 이를 기반으로 벡터 DB에서 가장 유사한 이미지를 검색합니다.
이후, 검색된 이미지를 DINOv2를 활용해 추가적인 이미지 유사도 검색을 수행하여 최종 결과를 제공합니다.



## 사용 기술
	•	
   -  OpenCLIP (텍스트-이미지 임베딩 생성)
	-  DINOv2 (이미지 임베딩 생성 및 유사도 검색)
	•	데이터베이스
	  - FAISS (벡터 데이터베이스, 빠른 유사도 검색)
	•	기타
	  - Python
	  - NumPy, OpenCV
	  - OS 및 파일 관리 관련 라이브러리
  
