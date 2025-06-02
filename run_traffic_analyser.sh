python traffic_analyser.py --detection-gpu-mem 0.5 \
--detection-model models/detectors/vehicle_peds_frcnn_res101/frozen_inference_graph.pb \
--classification-model models/classifiers/classifier_garage_ResNet50_2DBB \
--extractor-model models/feature_extractors/vehicle_MobileNet_AIC_128dim/ \
--output test_output \
 "rtsp://viewer:bozka@upgm-ipkam5.fit.vutbr.cz/axis-media/media.amp?resolution=1920x1080&bitrate=4000&fps=15"
