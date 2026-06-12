import cv2
import numpy as np
import os
import pyzed.sl as sl

class StereoZEDCalibrator:
    def __init__(self, pattern_size=(8, 5), square_size=20.0):
        # 1. Inicialización de la cámara ZED
        self.init_params = sl.InitParameters()
        self.init_params.camera_resolution = sl.RESOLUTION.HD720 # Resolución recomendada para calibrar
        self.init_params.camera_fps = 30
        
        self.cam = sl.Camera()
        if self.cam.open(self.init_params) != sl.ERROR_CODE.SUCCESS:
            print("Error al abrir la cámara ZED.")
            exit()
            
        self.runtime = sl.RuntimeParameters()
        self.mat_left = sl.Mat()
        self.mat_right = sl.Mat()
        
        # 2. Variables de calibración
        self.pattern_size = pattern_size
        self.square_size = square_size
        
        # Matrices intrínsecas y coeficientes de distorsión
        self.K_l, self.D_l = None, None
        self.K_r, self.D_r = None, None
        
        # Matrices estéreo (Rotación, Traslación, Esencial y Fundamental)
        self.R, self.T, self.E, self.F = None, None, None, None
        
        # Generar puntos 3D del mundo real (Z=0). 
        # mgrid crea una cuadrícula, reshape(-1,2) la aplana en una lista de coordenadas (x,y)
        self.objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
        self.objp *= self.square_size

        # Crear directorios para guardar las imágenes si no existen
        os.makedirs("./imgs/left", exist_ok=True)
        os.makedirs("./imgs/right", exist_ok=True)

    def capture_and_calibrate(self, num_images=15):
        objpoints = [] # Puntos 3D en el espacio
        imgpoints_l = [] # Puntos 2D de la cámara izquierda
        imgpoints_r = [] # Puntos 2D de la cámara derecha
        accepted_images = 0

        print(f"\n--- INICIO DE CAPTURA ESTÉREO ---")
        print(f"Buscando {num_images} pares de imágenes válidas.")
        print("Mueve el tablero (diferentes ángulos, distancias, bordes de la imagen).")
        print("Presiona 'c' para capturar. Presiona 'q' para salir.")

        while accepted_images < num_images:
            if self.cam.grab(self.runtime) == sl.ERROR_CODE.SUCCESS:
                self.cam.retrieve_image(self.mat_left, sl.VIEW.LEFT)
                self.cam.retrieve_image(self.mat_right, sl.VIEW.RIGHT)
                
                # ZED devuelve imágenes en formato BGRA (4 canales). Las pasamos a BGR (3 canales) para OpenCV.
                frame_l = cv2.cvtColor(self.mat_left.get_data(), cv2.COLOR_BGRA2BGR)
                frame_r = cv2.cvtColor(self.mat_right.get_data(), cv2.COLOR_BGRA2BGR)

                cv2.imshow('ZED Left', frame_l)
                cv2.imshow('ZED Right', frame_r)

                key = cv2.waitKey(5) & 0xFF
                if key == ord('q'):
                    break
                    
                # Proceso de captura al presionar 'c'
                elif key == ord('c'):
                    gray_l = cv2.cvtColor(frame_l, cv2.COLOR_BGR2GRAY)
                    gray_r = cv2.cvtColor(frame_r, cv2.COLOR_BGR2GRAY)

                    # Buscamos las esquinas en AMBAS imágenes a la vez
                    ret_l, corners_l = cv2.findChessboardCorners(gray_l, self.pattern_size, None)
                    ret_r, corners_r = cv2.findChessboardCorners(gray_r, self.pattern_size, None)

                    # REGLA DE ORO ESTÉREO: Solo nos sirve si el tablero se ve completo en ambas cámaras
                    if ret_l and ret_r:
                        # Refinamos las esquinas para precisión subpíxel
                        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                        corners_l = cv2.cornerSubPix(gray_l, corners_l, (11, 11), (-1, -1), criteria)
                        corners_r = cv2.cornerSubPix(gray_r, corners_r, (11, 11), (-1, -1), criteria)

                        # Dibujamos para confirmación visual
                        draw_l = frame_l.copy()
                        draw_r = frame_r.copy()
                        cv2.drawChessboardCorners(draw_l, self.pattern_size, corners_l, ret_l)
                        cv2.drawChessboardCorners(draw_r, self.pattern_size, corners_r, ret_r)
                        
                        cv2.imshow('Confirm L', draw_l)
                        cv2.imshow('Confirm R', draw_r)
                        print("\nPatrón detectado en ambas lentes. ¿Aceptar? (y/n)")

                        while True:
                            k = cv2.waitKey(0) & 0xFF
                            if k == ord('y'):
                                objpoints.append(self.objp)
                                imgpoints_l.append(corners_l)
                                imgpoints_r.append(corners_r)
                                
                                # Guardamos las imágenes físicas en la carpeta
                                cv2.imwrite(f"./imgs/left/calib_{accepted_images}.bmp", frame_l)
                                cv2.imwrite(f"./imgs/right/calib_{accepted_images}.bmp", frame_r)
                                
                                accepted_images += 1
                                print(f"Par guardado. ({accepted_images}/{num_images})")
                                break
                            elif k == ord('n'):
                                print("Par descartado.")
                                break
                                
                        cv2.destroyWindow('Confirm L')
                        cv2.destroyWindow('Confirm R')
                    else:
                        print("El patrón no se ve completamente en una o ambas cámaras. Muévelo.")

        cv2.destroyWindow('ZED Left')
        cv2.destroyWindow('ZED Right')

        if accepted_images == num_images:
            print("\nCalculando parámetros... Esto puede tardar unos segundos.")
            h, w = frame_l.shape[:2]
            img_shape = (w, h)

            # PASO 1: Calibrar Lente Izquierda individualmente
            _, self.K_l, self.D_l, _, _ = cv2.calibrateCamera(objpoints, imgpoints_l, img_shape, None, None)
            
            # PASO 2: Calibrar Lente Derecha individualmente
            _, self.K_r, self.D_r, _, _ = cv2.calibrateCamera(objpoints, imgpoints_r, img_shape, None, None)

            # PASO 3: Calibración Estéreo (Calcula la relación entre ambas cámaras)
            # Fijamos los parámetros intrínsecos (CALIB_FIX_INTRINSIC) porque ya los calculamos en los pasos 1 y 2.
            flags = cv2.CALIB_FIX_ASPECT_RATIO + cv2.CALIB_FIX_INTRINSIC
            criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-5)
            
            # El vector de traslacion T representa la distancia entre ambas camaras, gracias a las medidas de cada cuadrado del tablero que son 20mm
            ret_stereo, _, _, _, _, self.R, self.T, self.E, self.F = cv2.stereoCalibrate(
                objpoints, imgpoints_l, imgpoints_r, 
                self.K_l, self.D_l, self.K_r, self.D_r, 
                img_shape, criteria=criteria, flags=flags
            )

            print(f"¡Calibración Estéreo exitosa! Error RMS: {ret_stereo:.4f}")
            np.savez("stereo_calib.npz", Kl=self.K_l, Dl=self.D_l, Kr=self.K_r, Dr=self.D_r, R=self.R, T=self.T)
            return True
        return False

    def load_calibration(self, filename='stereo_calib.npz'):
        if os.path.exists(filename):
            data = np.load(filename)
            self.K_l, self.D_l = data['Kl'], data['Dl']
            self.K_r, self.D_r = data['Kr'], data['Dr']
            self.R, self.T = data['R'], data['T']
            print("\n¡Parámetros cargados desde archivo!")
            return True
        return False

    def project_virtual_point(self):
        if self.K_l is None or self.K_r is None:
            print("Error: Primero debes calibrar el sistema estéreo.")
            return

        print("\n--- INICIO DE PROYECCIÓN 3D ESTÉREO ---")
        print("Mostrando proyecciones en ambas lentes... Presiona 'q' para salir.")

        # Calculamos el centro geométrico del tablero para dibujar el cubo allí
        cx = (self.pattern_size[0] - 1) / 2.0
        cy = (self.pattern_size[1] - 1) / 2.0
        l = 1.5 # Medio ancho del cubo

        # Definimos los 8 puntos de un cubo 3D virtual
        axis = np.float32([
            [cx-l, cy-l, 0],  [cx-l, cy+l, 0],  [cx+l, cy+l, 0],  [cx+l, cy-l, 0],     # Base
            [cx-l, cy-l, -3], [cx-l, cy+l, -3], [cx+l, cy+l, -3], [cx+l, cy-l, -3]     # Techo
        ]) * self.square_size

        while True:
            if self.cam.grab(self.runtime) == sl.ERROR_CODE.SUCCESS:
                # 1. Recuperamos las imágenes de AMBAS lentes
                self.cam.retrieve_image(self.mat_left, sl.VIEW.LEFT)
                self.cam.retrieve_image(self.mat_right, sl.VIEW.RIGHT)
                
                frame_l = cv2.cvtColor(self.mat_left.get_data(), cv2.COLOR_BGRA2BGR)
                frame_r = cv2.cvtColor(self.mat_right.get_data(), cv2.COLOR_BGRA2BGR)
                
                gray_l = cv2.cvtColor(frame_l, cv2.COLOR_BGR2GRAY)
                gray_r = cv2.cvtColor(frame_r, cv2.COLOR_BGR2GRAY)

                # 2. Buscamos el tablero en AMBAS imágenes
                found_l, corners_l = cv2.findChessboardCorners(gray_l, self.pattern_size, None)
                found_r, corners_r = cv2.findChessboardCorners(gray_r, self.pattern_size, None)

                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

                # ==========================================
                # PROCESAMIENTO CÁMARA IZQUIERDA
                # ==========================================
                if found_l:
                    corners_l = cv2.cornerSubPix(gray_l, corners_l, (11, 11), (-1, -1), criteria)
                    # Usamos K_l y D_l (parámetros de la lente izquierda)
                    ret_pnp_l, rvec_l, tvec_l = cv2.solvePnP(self.objp, corners_l, self.K_l, self.D_l)
                    
                    if ret_pnp_l:
                        imgpts_l, _ = cv2.projectPoints(axis, rvec_l, tvec_l, self.K_l, self.D_l)
                        imgpts_l = np.int32(imgpts_l).reshape(-1, 2)

                        frame_l = cv2.drawContours(frame_l, [imgpts_l[:4]], -1, (0, 255, 0), -3) # Base
                        for i, j in zip(range(4), range(4,8)):
                            frame_l = cv2.line(frame_l, tuple(imgpts_l[i]), tuple(imgpts_l[j]), (255, 0, 0), 3) # Pilares
                        frame_l = cv2.drawContours(frame_l, [imgpts_l[4:]], -1, (0, 0, 255), 3) # Techo

                # ==========================================
                # PROCESAMIENTO CÁMARA DERECHA
                # ==========================================
                if found_r:
                    corners_r = cv2.cornerSubPix(gray_r, corners_r, (11, 11), (-1, -1), criteria)
                    # ¡AQUÍ ESTÁ LA CLAVE! Usamos K_r y D_r (parámetros de la lente derecha)
                    ret_pnp_r, rvec_r, tvec_r = cv2.solvePnP(self.objp, corners_r, self.K_r, self.D_r)
                    
                    if ret_pnp_r:
                        imgpts_r, _ = cv2.projectPoints(axis, rvec_r, tvec_r, self.K_r, self.D_r)
                        imgpts_r = np.int32(imgpts_r).reshape(-1, 2)

                        frame_r = cv2.drawContours(frame_r, [imgpts_r[:4]], -1, (0, 255, 0), -3) # Base
                        for i, j in zip(range(4), range(4,8)):
                            frame_r = cv2.line(frame_r, tuple(imgpts_r[i]), tuple(imgpts_r[j]), (255, 0, 0), 3) # Pilares
                        frame_r = cv2.drawContours(frame_r, [imgpts_r[4:]], -1, (0, 0, 255), 3) # Techo

                # 3. Mostramos ambas ventanas sincronizadas
                cv2.imshow('Realidad Aumentada - ZED Izquierda', frame_l)
                cv2.imshow('Realidad Aumentada - ZED Derecha', frame_r)

            if cv2.waitKey(5) & 0xFF == ord('q'):
                break
                
        cv2.destroyWindow('Realidad Aumentada - ZED Izquierda')
        cv2.destroyWindow('Realidad Aumentada - ZED Derecha')

    def close(self):
        self.cam.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    # Asegúrate de poner el tamaño correcto de tu tablero (puntos internos)
    # OpenCV no sabe las unidades de cada cuadrado (en nuestro caso son cada uno de 20mm)
    calibrador = StereoZEDCalibrator(pattern_size=(8, 5), square_size=20.0)
    
    ARCHIVO = 'stereo_calib.npz'
    
    if os.path.exists(ARCHIVO):
        resp = input(f"Se encontró '{ARCHIVO}'. ¿Cargar? (s/n): ").strip().lower()
        if resp == 's':
            calibrado = calibrador.load_calibration(ARCHIVO)
        else:
            calibrado = calibrador.capture_and_calibrate(num_images=15)
    else:
        calibrado = calibrador.capture_and_calibrate(num_images=15)
        
    if calibrado:
        calibrador.project_virtual_point()
        
    calibrador.close()