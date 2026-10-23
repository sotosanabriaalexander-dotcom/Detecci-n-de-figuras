import cv2
import numpy as np
import os

def detectar_figuras(frame):
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    edges = cv2.Canny(blurred, 50, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for cnt in contours:
        if cv2.contourArea(cnt) < 300:
            continue
        
        approx = cv2.approxPolyDP(cnt, 0.04 * cv2.arcLength(cnt, True), True)
        
        x, y, w, h = cv2.boundingRect(approx)
        
        num_lados = len(approx)
        if num_lados == 3:
            figura = "Triángulo"
        elif num_lados == 4:
            aspect_ratio = w / float(h)
            figura = "Cuadrado" if 0.95 <= aspect_ratio <= 1.05 else "Rectángulo"
        elif num_lados == 5:
            figura = "Pentágono"
        elif num_lados == 6:
            figura = "Hexágono"
        else:
            figura = "Círculo"
        
        cv2.drawContours(frame, [approx], -1, (0, 255, 0), 2)
        cv2.putText(frame, figura, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    return frame

def main():
    ruta_imagen = "Samus.jpg"
    
    if not os.path.exists(ruta_imagen):
        print(f"Error: No se encontró el archivo '{ruta_imagen}'.")
        return

    imagen = cv2.imread(ruta_imagen)
    if imagen is None:
        print("Error: No se pudo leer la imagen.")
        return
    
    imagen_con_figuras = detectar_figuras(imagen)
    
    cv2.imshow("Reconocimiento de Figuras Geometricas", imagen_con_figuras)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
