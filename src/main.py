from .config import FLAGS, CFG
from .train_teacher import train_teacher_step
from .inference import ClassifierService
from .utils import LOG

def main():
    if FLAGS.train_teacher:
        LOG.info("Starting teacher training...")
        train_teacher_step()

    if FLAGS.infer_demo:
        LOG.info("Running demo inference...")
        service = ClassifierService(f"{CFG.out_dir}/onnx/teacher_fp32.onnx", CFG.teacher_model)
        sample = ["This is a test message"]
        pred = service.predict(sample)
        LOG.info("Demo prediction: %s", pred)

if __name__ == "__main__":
    main()
