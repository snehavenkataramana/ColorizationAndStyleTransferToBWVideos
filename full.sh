python test.py --name siggraph_reg2

rm examples/inputs/content_images/* || true

cp results/siggraph_reg2/val_latest/images/*fake_reg.png examples/inputs/content_images

python neural_style.py -style_image examples/inputs/cartoon_ref.png -content_image examples/inputs/content_images -num_iterations 200 -image_size 256 -content_weight 50 -style_weight 100
