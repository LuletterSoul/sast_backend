
import cv2
import os


def unframe(input_path, output_path):
    cap = cv2.VideoCapture(input_path)
    cnt = 0
    os.makedirs(output_path, exist_ok=True)
    ret, frame = cap.read()
    while ret:
        if cnt == 0:
            cv2.imwrite(os.path.join(output_path, 'preview.png'), frame)
        cv2.imwrite(os.path.join(output_path, f'{cnt}.png'), frame)
        cnt = cnt + 1
        ret, frame = cap.read()
        print(f'Unframed {cnt} frames')


if __name__ == "__main__":
    video_dir = 'data/contents/Video'
    output_dir = 'data/contents/Video'

    video_names = os.listdir(video_dir)

    for v in video_names:
        video_path = os.path.join(video_dir, v)
        video_prefix = os.path.splitext(v)[0]
        output_path = os.path.join(video_dir, video_prefix)
        print(f'Processing {video_path}')
        unframe(video_path, output_path)
        print(f'Done {video_path}')
