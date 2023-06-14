import triton_python_backend_utils as pb_utils
import numpy as np
from tensorflow.keras.models import load_model

class TritonPythonModel:
    def initialize(self, args):
        """모델 초기화 함수"""
        print('Initialized...')
        self.loaded_model = load_model('posneg_no_oneword_v3.h5')

    def execute(self, requests):
        responses = []

        for request in requests:
            input_tensor = pb_utils.get_input_tensor_by_name(request, 'input_tensor')
            input_data = input_tensor.as_numpy()

            # 모델 추론 실행
            output_data = self.loaded_model.predict(input_data)

            # 결과를 새로운 텐서로 저장
            output_tensor = pb_utils.Tensor(output_data)

            # 추론 응답 생성
            response = pb_utils.InferenceResponse(output_tensors=[output_tensor])
            responses.append(response)

        return responses


    def finalize(self):
        """모델 종료 함수"""
        print('Cleaning up...')

# 모델 객체 생성
model = TritonPythonModel()

# 모델을 Triton Inference Server에 등록
pb_utils.run(model)
