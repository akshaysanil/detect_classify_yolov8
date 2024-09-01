from bing_image_downloader import downloader


downloader.download('gym equipment band', limit=400, output_dir='dataset',
                     adult_filter_off=False, force_replace=False, timeout=60, verbose=True)

