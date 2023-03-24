How to use it

```py
data_dir = 'dataset'
data = load_image_dataset(data_dir)
data_iterator = data.as_numpy_iterator()
show_image_batch(data_iterator, num_images=4, figsize=(20,20))
```