FROM public.ecr.aws/lambda/python:3.12

# OpenCV needs libGL; Lambda's filesystem is read-only except /tmp
RUN dnf install -y mesa-libGL && dnf clean all
ENV HOME=/tmp MPLCONFIGDIR=/tmp

COPY requirements-cloud.txt .
RUN pip install --no-cache-dir -r requirements-cloud.txt

COPY ego2sim/ ${LAMBDA_TASK_ROOT}/ego2sim/
COPY models/hand_landmarker.task ${LAMBDA_TASK_ROOT}/models/

CMD ["ego2sim.vision.lambda_handler.handler"]
