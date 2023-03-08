from modules import shared


class FaceRestoration:
    """
    脸部修复接口. 目前实现有 FaceRestorerCodeFormer, FaceRestorerGFPGAN
    """
    def name(self):
        return "None"

    def restore(self, np_image):
        return np_image


def restore_faces(np_image):
    """
    脸部修复. 遍历shared.face_restorers找出一个合适的来用
    """
    face_restorers = [x for x in shared.face_restorers if x.name() == shared.opts.face_restoration_model or shared.opts.face_restoration_model is None]
    if len(face_restorers) == 0:
        return np_image

    face_restorer = face_restorers[0]

    return face_restorer.restore(np_image)
