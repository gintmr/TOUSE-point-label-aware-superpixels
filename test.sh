python propagate.py -r ./test/images -g ./test/jsons -l ./save -p ./save --ensemble --num_labels 100

python propagate.py -r ./1353_1020/images -g ./1353_1020/jsons -l ./save -p ./save -y 1020 -w 1353 --ensemble --num_labels 100

python propagate.py -r ./try/row -g ./try/instance -l ./save -p ./save -y 1024 -w 2048 --ensemble --num_labels 50 --device gpu

python propagate.py -r ./superpixe_testing/new_images -g ./superpixe_testing/mask_imgs -l ./save -p ./save -y 720 -w 720 --ensemble --points --device cpu

python propagate.py -r ./test11/new_images -g ./test11/mask_imgs -l ./save -p ./save -y 720 -w 720 --ensemble --num_labels 500 --c 256 --device gpu

python propagate.py -r ./test/img -g ./test/mask -json ./test/json -l ./save -p ./save -y 720 -w 720 --ensemble --c 256 --points --device gpu