import cv2
import numpy as np
from ultralytics import YOLO
from flask import Flask, jsonify
from flask import request, redirect
import threading
import time
from datetime import datetime
import qrcode
import uuid 
import os

# ==========================================
# 1. CONFIGURACIÓN
# ==========================================
VIDEO_PATH = "C:/Users/choco/Documents/Python/Labview_Estacionamiento/Parking0.5.mp4"
MODEL_PATH = "C:/Users/choco/Downloads/best.pt"
OUTPUT_PATH = "C:/Users/choco/Documents/Python/Labview_Estacionamiento/Parking_Salida.mp4"
IMAGEN_SALIDA = "C:/Users/choco/Documents/Python/Labview_Estacionamiento/frame_actual.jpg"  

UMBRAL_CONFIANZA = 0.05 
VELOCIDAD_REPRODUCCION = 10

# CAJONES 
LUGARES = {
    "1": np.array([[827, 333], [827, 423], [863, 423], [863, 333]], np.int32),
    "2": np.array([[863, 333], [863, 423], [899, 423], [899, 333]], np.int32),
    "3": np.array([[899, 333], [899, 423], [935, 423], [935, 333]], np.int32),
    "4": np.array([[935, 333], [935, 423], [971, 423], [971, 333]], np.int32),
    "5": np.array([[971, 333], [971, 423], [1007, 423], [1007, 333]], np.int32),
    "6": np.array([[827, 423], [827, 513], [863, 513], [863, 423]], np.int32),
    "7": np.array([[863, 423], [863, 513], [899, 513], [899, 423]], np.int32),
    "8": np.array([[899, 423], [899, 513], [935, 513], [935, 423]], np.int32),
    "9": np.array([[935, 423], [935, 513], [971, 513], [971, 423]], np.int32),
    "10": np.array([[971, 423], [971, 513], [1007, 513], [1007, 423]], np.int32),
    "11": np.array([[1007, 333], [1007, 423], [1042, 423], [1042, 333]], np.int32),
    "12": np.array([[1042, 333], [1042, 423], [1077, 423], [1077, 333]], np.int32),
    "13": np.array([[1077, 333], [1077, 423], [1112, 423], [1112, 333]], np.int32),
    "14": np.array([[1112, 333], [1112, 423], [1147, 423], [1147, 333]], np.int32),
    "15": np.array([[1147, 333], [1147, 423], [1182, 423], [1182, 333]], np.int32),
    "16": np.array([[1007, 423], [1007, 513], [1042, 513], [1042, 423]], np.int32),
    "17": np.array([[1042, 423], [1042, 513], [1077, 513], [1077, 423]], np.int32),
    "18": np.array([[1077, 423], [1077, 513], [1112, 513], [1112, 423]], np.int32),
    "19": np.array([[1112, 423], [1112, 513], [1147, 513], [1147, 423]], np.int32),
    "20": np.array([[1147, 423], [1147, 513], [1182, 513], [1182, 423]], np.int32),
    "21": np.array([[1182, 333], [1182, 423], [1219, 423], [1219, 333]], np.int32),
    "22": np.array([[1219, 333], [1219, 423], [1256, 423], [1256, 333]], np.int32),
    "23": np.array([[1256, 333], [1256, 423], [1293, 423], [1293, 333]], np.int32),
    "24": np.array([[1293, 333], [1293, 423], [1330, 423], [1330, 333]], np.int32),
    "25": np.array([[1330, 333], [1330, 423], [1367, 423], [1367, 333]], np.int32),
    "26": np.array([[1182, 423], [1182, 513], [1219, 513], [1219, 423]], np.int32),
    "27": np.array([[1219, 423], [1219, 513], [1256, 513], [1256, 423]], np.int32),
    "28": np.array([[1256, 423], [1256, 513], [1293, 513], [1293, 423]], np.int32),
    "29": np.array([[1293, 423], [1293, 513], [1330, 513], [1330, 423]], np.int32),
    "30": np.array([[1330, 423], [1330, 513], [1367, 513], [1367, 423]], np.int32),
    "31": np.array([[1367, 333], [1367, 423], [1404, 423], [1404, 333]], np.int32),
    "32": np.array([[1404, 333], [1404, 423], [1441, 423], [1441, 333]], np.int32),
    "33": np.array([[1441, 333], [1441, 423], [1478, 423], [1478, 333]], np.int32),
    "34": np.array([[1478, 333], [1478, 423], [1515, 423], [1515, 333]], np.int32),
    "35": np.array([[1515, 333], [1515, 423], [1552, 423], [1552, 333]], np.int32),
    "36": np.array([[1367, 423], [1367, 513], [1404, 513], [1404, 423]], np.int32),
    "37": np.array([[1404, 423], [1404, 513], [1441, 513], [1441, 423]], np.int32),
    "38": np.array([[1441, 423], [1441, 513], [1478, 513], [1478, 423]], np.int32),
    "39": np.array([[1478, 423], [1478, 513], [1515, 513], [1515, 423]], np.int32),
    "40": np.array([[1515, 423], [1515, 513], [1552, 513], [1552, 423]], np.int32),
    "41": np.array([[790, 333], [790, 423], [826, 423], [826, 333]], np.int32),
    "42": np.array([[754, 333], [754, 423], [790, 423], [790, 333]], np.int32),
    "43": np.array([[718, 333], [718, 423], [754, 423], [754, 333]], np.int32),
    "44": np.array([[682, 333], [682, 423], [718, 423], [718, 333]], np.int32),
    "45": np.array([[646, 333], [646, 423], [682, 423], [682, 333]], np.int32),
    "46": np.array([[790, 423], [790, 513], [826, 513], [826, 423]], np.int32),
    "47": np.array([[754, 423], [754, 513], [790, 513], [790, 423]], np.int32),
    "48": np.array([[718, 423], [718, 513], [754, 513], [754, 423]], np.int32),
    "49": np.array([[682, 423], [682, 513], [718, 513], [718, 423]], np.int32),
    "50": np.array([[646, 423], [646, 513], [682, 513], [682, 423]], np.int32),
    "51": np.array([[609, 333], [609, 423], [646, 423], [646, 333]], np.int32),
    "52": np.array([[572, 333], [572, 423], [609, 423], [609, 333]], np.int32),
    "53": np.array([[535, 333], [535, 423], [572, 423], [572, 333]], np.int32),
    "54": np.array([[498, 333], [498, 423], [535, 423], [535, 333]], np.int32),
    "55": np.array([[461, 333], [461, 423], [498, 423], [498, 333]], np.int32),
    "56": np.array([[609, 423], [609, 513], [646, 513], [646, 423]], np.int32),
    "57": np.array([[572, 423], [572, 513], [609, 513], [609, 423]], np.int32),
    "58": np.array([[535, 423], [535, 513], [572, 513], [572, 423]], np.int32),
    "59": np.array([[498, 423], [498, 513], [535, 513], [535, 423]], np.int32),
    "60": np.array([[461, 423], [461, 513], [498, 513], [498, 423]], np.int32),
    "61": np.array([[425, 333], [425, 423], [461, 423], [461, 333]], np.int32),
    "62": np.array([[389, 333], [389, 423], [425, 423], [425, 333]], np.int32),
    "63": np.array([[353, 333], [353, 423], [389, 423], [389, 333]], np.int32),
    "64": np.array([[317, 333], [317, 423], [353, 423], [353, 333]], np.int32),
    "65": np.array([[281, 333], [281, 423], [317, 423], [317, 333]], np.int32),
    "66": np.array([[425, 423], [425, 513], [461, 513], [461, 423]], np.int32),
    "67": np.array([[389, 423], [389, 513], [425, 513], [425, 423]], np.int32),
    "68": np.array([[353, 423], [353, 513], [389, 513], [389, 423]], np.int32),
    "69": np.array([[317, 423], [317, 513], [353, 513], [353, 423]], np.int32),
    "70": np.array([[281, 423], [281, 513], [317, 513], [317, 423]], np.int32),
    "71": np.array([[246, 333], [246, 423], [281, 423], [281, 333]], np.int32),
    "72": np.array([[246, 423], [246, 513], [281, 513], [281, 423]], np.int32),
    "73": np.array([[563, 600], [563, 690], [600, 690], [600, 600]], np.int32),
    "74": np.array([[600, 600], [600, 690], [637, 690], [637, 600]], np.int32),
    "75": np.array([[637, 600], [637, 690], [674, 690], [674, 600]], np.int32),
    "76": np.array([[674, 600], [674, 690], [711, 690], [711, 600]], np.int32),
    "78": np.array([[563, 690], [563, 780], [600, 780], [600, 690]], np.int32),
    "79": np.array([[600, 690], [600, 780], [637, 780], [637, 690]], np.int32),
    "80": np.array([[637, 690], [637, 780], [674, 780], [674, 690]], np.int32),
    "81": np.array([[674, 690], [674, 780], [711, 780], [711, 690]], np.int32),
    "83": np.array([[711, 600], [711, 690], [747, 690], [747, 600]], np.int32),
    "84": np.array([[747, 600], [747, 690], [783, 690], [783, 600]], np.int32),
    "85": np.array([[783, 600], [783, 690], [819, 690], [819, 600]], np.int32),
    "86": np.array([[819, 600], [819, 690], [855, 690], [855, 600]], np.int32),
    "88": np.array([[711, 690], [711, 780], [747, 780], [747, 690]], np.int32),
    "89": np.array([[747, 690], [747, 780], [783, 780], [783, 690]], np.int32),
    "90": np.array([[783, 690], [783, 780], [819, 780], [819, 690]], np.int32),
    "91": np.array([[819, 690], [819, 780], [855, 780], [855, 690]], np.int32),
    "93": np.array([[855, 600], [855, 690], [891, 690], [891, 600]], np.int32),
    "94": np.array([[891, 600], [891, 690], [927, 690], [927, 600]], np.int32),
    "95": np.array([[927, 600], [927, 690], [963, 690], [963, 600]], np.int32),
    "96": np.array([[963, 600], [963, 690], [999, 690], [999, 600]], np.int32),
    "97": np.array([[999, 600], [999, 690], [1035, 690], [1035, 600]], np.int32),
    "98": np.array([[855, 690], [855, 780], [891, 780], [891, 690]], np.int32),
    "99": np.array([[891, 690], [891, 780], [927, 780], [927, 690]], np.int32),
    "100": np.array([[927, 690], [927, 780], [963, 780], [963, 690]], np.int32),
    "101": np.array([[963, 690], [963, 780], [999, 780], [999, 690]], np.int32),
    "102": np.array([[999, 690], [999, 780], [1035, 780], [1035, 690]], np.int32),
    "103": np.array([[526, 600], [526, 690], [563, 690], [563, 600]], np.int32),
    "104": np.array([[489, 600], [489, 690], [526, 690], [526, 600]], np.int32),
    "105": np.array([[452, 600], [452, 690], [489, 690], [489, 600]], np.int32),
    "106": np.array([[415, 600], [415, 690], [452, 690], [452, 600]], np.int32),
    "107": np.array([[378, 600], [378, 690], [415, 690], [415, 600]], np.int32),
    "108": np.array([[526, 690], [526, 780], [563, 780], [563, 690]], np.int32),
    "109": np.array([[489, 690], [489, 780], [526, 780], [526, 690]], np.int32),
    "110": np.array([[452, 690], [452, 780], [489, 780], [489, 690]], np.int32),
    "111": np.array([[415, 690], [415, 780], [452, 780], [452, 690]], np.int32),
    "112": np.array([[378, 690], [378, 780], [415, 780], [415, 690]], np.int32),
    "113": np.array([[341, 600], [341, 690], [378, 690], [378, 600]], np.int32),
    "114": np.array([[304, 600], [304, 690], [341, 690], [341, 600]], np.int32),
    "115": np.array([[267, 600], [267, 690], [304, 690], [304, 600]], np.int32),
    "116": np.array([[230, 600], [230, 690], [267, 690], [267, 600]], np.int32),
    "117": np.array([[341, 690], [341, 780], [378, 780], [378, 690]], np.int32),
    "118": np.array([[304, 690], [304, 780], [341, 780], [341, 690]], np.int32),
    "119": np.array([[267, 690], [267, 780], [304, 780], [304, 690]], np.int32),
    "120": np.array([[230, 690], [230, 780], [267, 780], [267, 690]], np.int32),
    "121": np.array([[1109, 597], [1109, 687], [1146, 687], [1146, 597]], np.int32),
    "122": np.array([[1146, 597], [1146, 687], [1183, 687], [1183, 597]], np.int32),
    "123": np.array([[1183, 597], [1183, 687], [1220, 687], [1220, 597]], np.int32),
    "124": np.array([[1220, 597], [1220, 687], [1257, 687], [1257, 597]], np.int32),
    "125": np.array([[1257, 597], [1257, 687], [1294, 687], [1294, 597]], np.int32),
    "126": np.array([[1294, 597], [1294, 687], [1331, 687], [1331, 597]], np.int32),
    "127": np.array([[1331, 597], [1331, 687], [1368, 687], [1368, 597]], np.int32),
    "128": np.array([[1368, 597], [1368, 687], [1405, 687], [1405, 597]], np.int32),
    "129": np.array([[1405, 597], [1405, 687], [1442, 687], [1442, 597]], np.int32),
    "130": np.array([[1442, 597], [1442, 687], [1479, 687], [1479, 597]], np.int32),
    "131": np.array([[1479, 597], [1479, 687], [1515, 687], [1515, 597]], np.int32),
    "132": np.array([[1515, 597], [1515, 687], [1551, 687], [1551, 597]], np.int32),
    "133": np.array([[1551, 597], [1551, 687], [1587, 687], [1587, 597]], np.int32),
    "134": np.array([[1587, 597], [1587, 687], [1623, 687], [1623, 597]], np.int32),
    "135": np.array([[1623, 597], [1623, 687], [1659, 687], [1659, 597]], np.int32),
    "136": np.array([[1659, 597], [1659, 687], [1695, 687], [1695, 597]], np.int32),
    "137": np.array([[1695, 597], [1695, 687], [1731, 687], [1731, 597]], np.int32),
    "138": np.array([[1731, 597], [1731, 687], [1767, 687], [1767, 597]], np.int32),
    "139": np.array([[1442, 687], [1442, 777], [1478, 777], [1478, 687]], np.int32),
    "140": np.array([[1478, 687], [1478, 777], [1514, 777], [1514, 687]], np.int32),
    "141": np.array([[1514, 687], [1514, 777], [1550, 777], [1550, 687]], np.int32),
    "142": np.array([[1550, 687], [1550, 777], [1586, 777], [1586, 687]], np.int32),
    "143": np.array([[1586, 687], [1586, 777], [1622, 777], [1622, 687]], np.int32),
    "144": np.array([[1622, 687], [1622, 777], [1658, 777], [1658, 687]], np.int32),
    "145": np.array([[1658, 687], [1658, 777], [1694, 777], [1694, 687]], np.int32),
    "146": np.array([[1694, 687], [1694, 777], [1730, 777], [1730, 687]], np.int32),
    "147": np.array([[1730, 687], [1730, 777], [1766, 777], [1766, 687]], np.int32),
    "148": np.array([[407, 868], [407, 958], [449, 958], [449, 868]], np.int32),
    "149": np.array([[449, 868], [449, 958], [487, 958], [487, 868]], np.int32),
    "150": np.array([[487, 868], [487, 958], [523, 958], [523, 868]], np.int32),
    "151": np.array([[523, 868], [523, 958], [559, 958], [559, 868]], np.int32),
    "152": np.array([[559, 868], [559, 958], [595, 958], [595, 868]], np.int32),
    "153": np.array([[595, 868], [595, 958], [631, 958], [631, 868]], np.int32),
    "154": np.array([[631, 868], [631, 958], [667, 958], [667, 868]], np.int32),
    "155": np.array([[487, 958], [487, 1048], [523, 1048], [523, 958]], np.int32),
    "156": np.array([[523, 958], [523, 1048], [559, 1048], [559, 958]], np.int32),
    "157": np.array([[559, 958], [559, 1048], [595, 1048], [595, 958]], np.int32),
    "158": np.array([[595, 958], [595, 1048], [631, 1048], [631, 958]], np.int32),
    "159": np.array([[631, 958], [631, 1048], [667, 1048], [667, 958]], np.int32),
    "160": np.array([[667, 868], [667, 958], [703, 958], [703, 868]], np.int32),
    "161": np.array([[703, 868], [703, 958], [739, 958], [739, 868]], np.int32),
    "162": np.array([[739, 868], [739, 958], [775, 958], [775, 868]], np.int32),
    "163": np.array([[775, 868], [775, 958], [811, 958], [811, 868]], np.int32),
    "164": np.array([[811, 868], [811, 958], [847, 958], [847, 868]], np.int32),
    "165": np.array([[667, 958], [667, 1048], [703, 1048], [703, 958]], np.int32),
    "166": np.array([[703, 958], [703, 1048], [739, 1048], [739, 958]], np.int32),
    "167": np.array([[739, 958], [739, 1048], [775, 1048], [775, 958]], np.int32),
    "168": np.array([[775, 958], [775, 1048], [811, 1048], [811, 958]], np.int32),
    "169": np.array([[811, 958], [811, 1048], [847, 1048], [847, 958]], np.int32),
    "170": np.array([[847, 868], [847, 958], [884, 958], [884, 868]], np.int32),
    "171": np.array([[884, 868], [884, 958], [921, 958], [921, 868]], np.int32),
    "172": np.array([[921, 868], [921, 958], [958, 958], [958, 868]], np.int32),
    "173": np.array([[958, 868], [958, 958], [995, 958], [995, 868]], np.int32),
    "174": np.array([[995, 868], [995, 958], [1032, 958], [1032, 868]], np.int32),
    "175": np.array([[847, 958], [847, 1048], [884, 1048], [884, 958]], np.int32),
    "176": np.array([[884, 958], [884, 1048], [921, 1048], [921, 958]], np.int32),
    "177": np.array([[921, 958], [921, 1048], [958, 1048], [958, 958]], np.int32),
    "178": np.array([[958, 958], [958, 1048], [995, 1048], [995, 958]], np.int32),
    "179": np.array([[995, 958], [995, 1048], [1032, 1048], [1032, 958]], np.int32),
    "180": np.array([[1032, 868], [1032, 958], [1069, 958], [1069, 868]], np.int32),
    "181": np.array([[1069, 868], [1069, 958], [1106, 958], [1106, 868]], np.int32),
    "182": np.array([[1032, 958], [1032, 1048], [1069, 1048], [1069, 958]], np.int32),
    "183": np.array([[1069, 958], [1069, 1048], [1106, 1048], [1106, 958]], np.int32),
    "184": np.array([[1106, 958], [1106, 1048], [1143, 1048], [1143, 958]], np.int32),
    "185": np.array([[1143, 958], [1143, 1048], [1180, 1048], [1180, 958]], np.int32),
    "186": np.array([[1180, 958], [1180, 1048], [1217, 1048], [1217, 958]], np.int32),
    "187": np.array([[1217, 958], [1217, 1048], [1254, 1048], [1254, 958]], np.int32),
    "188": np.array([[1254, 958], [1254, 1048], [1291, 1048], [1291, 958]], np.int32),
    "189": np.array([[1291, 958], [1291, 1048], [1328, 1048], [1328, 958]], np.int32),
    "190": np.array([[1328, 958], [1328, 1048], [1365, 1048], [1365, 958]], np.int32),
    "191": np.array([[1365, 958], [1365, 1048], [1402, 1048], [1402, 958]], np.int32),
    "192": np.array([[1402, 958], [1402, 1048], [1439, 1048], [1439, 958]], np.int32),
    "193": np.array([[1439, 868], [1439, 958], [1476, 958], [1476, 868]], np.int32),
    "194": np.array([[1476, 868], [1476, 958], [1513, 958], [1513, 868]], np.int32),
    "195": np.array([[1513, 868], [1513, 958], [1550, 958], [1550, 868]], np.int32),
    "196": np.array([[1550, 868], [1550, 958], [1587, 958], [1587, 868]], np.int32),
    "197": np.array([[1587, 868], [1587, 958], [1624, 958], [1624, 868]], np.int32),
    "198": np.array([[1439, 958], [1439, 1048], [1476, 1048], [1476, 958]], np.int32),
    "199": np.array([[1476, 958], [1476, 1048], [1513, 1048], [1513, 958]], np.int32),
    "200": np.array([[1513, 958], [1513, 1048], [1550, 1048], [1550, 958]], np.int32),
    "201": np.array([[1550, 958], [1550, 1048], [1587, 1048], [1587, 958]], np.int32),
    "202": np.array([[1587, 958], [1587, 1048], [1624, 1048], [1624, 958]], np.int32),
    "203": np.array([[1624, 868], [1624, 958], [1660, 958], [1660, 868]], np.int32),
    "204": np.array([[1660, 868], [1660, 958], [1696, 958], [1696, 868]], np.int32),
    "205": np.array([[1696, 868], [1696, 958], [1732, 958], [1732, 868]], np.int32),
    "206": np.array([[1732, 868], [1732, 958], [1768, 958], [1768, 868]], np.int32),
    "207": np.array([[1624, 958], [1624, 1048], [1660, 1048], [1660, 958]], np.int32),
    "208": np.array([[1660, 958], [1660, 1048], [1696, 1048], [1696, 958]], np.int32),
    "209": np.array([[1696, 958], [1696, 1048], [1732, 1048], [1732, 958]], np.int32),
    "210": np.array([[1732, 958], [1732, 1048], [1768, 1048], [1768, 958]], np.int32),
    "211": np.array([[920, 154], [920, 244], [957, 244], [957, 154]], np.int32),
    "212": np.array([[957, 154], [957, 244], [994, 244], [994, 154]], np.int32),
    "213": np.array([[994, 154], [994, 244], [1031, 244], [1031, 154]], np.int32),
    "214": np.array([[1031, 154], [1031, 244], [1068, 244], [1068, 154]], np.int32),
    "215": np.array([[1068, 154], [1068, 244], [1105, 244], [1105, 154]], np.int32),
    "216": np.array([[1105, 154], [1105, 244], [1142, 244], [1142, 154]], np.int32),
    "217": np.array([[1142, 154], [1142, 244], [1179, 244], [1179, 154]], np.int32),
    "218": np.array([[1179, 154], [1179, 244], [1216, 244], [1216, 154]], np.int32),
    "219": np.array([[1216, 154], [1216, 244], [1253, 244], [1253, 154]], np.int32),
    "220": np.array([[1253, 154], [1253, 244], [1288, 244], [1288, 154]], np.int32),
    "221": np.array([[1288, 154], [1288, 244], [1323, 244], [1323, 154]], np.int32),
    "222": np.array([[1323, 154], [1323, 244], [1358, 244], [1358, 154]], np.int32),
    "223": np.array([[1358, 154], [1358, 244], [1393, 244], [1393, 154]], np.int32),
    "224": np.array([[1393, 154], [1393, 244], [1428, 244], [1428, 154]], np.int32),
    "225": np.array([[1428, 154], [1428, 244], [1465, 244], [1465, 154]], np.int32),
    "226": np.array([[1465, 154], [1465, 244], [1502, 244], [1502, 154]], np.int32),
    "227": np.array([[1502, 154], [1502, 244], [1539, 244], [1539, 154]], np.int32),
    "228": np.array([[1539, 154], [1539, 244], [1576, 244], [1576, 154]], np.int32),
    "229": np.array([[1662, 251], [1662, 291], [1740, 291], [1740, 251]], np.int32),
    "230": np.array([[1662, 291], [1662, 331], [1740, 331], [1740, 291]], np.int32),
    "231": np.array([[1662, 331], [1662, 371], [1740, 371], [1740, 331]], np.int32),
    "232": np.array([[1662, 371], [1662, 411], [1740, 411], [1740, 371]], np.int32),
    "233": np.array([[1655, 432], [1655, 522], [1702, 522], [1702, 432]], np.int32),
    "234": np.array([[1702, 432], [1702, 522], [1739, 522], [1739, 432]], np.int32),
    "235": np.array([[1739, 432], [1739, 522], [1776, 522], [1776, 432]], np.int32),
    "236": np.array([[1776, 432], [1776, 522], [1813, 522], [1813, 432]], np.int32),
    "237": np.array([[1813, 432], [1813, 522], [1850, 522], [1850, 432]], np.int32),
    "238": np.array([[1850, 432], [1850, 522], [1887, 522], [1887, 432]], np.int32),
    "239": np.array([[1872, 544], [1872, 581], [1922, 581], [1922, 544]], np.int32),
    "240": np.array([[1872, 581], [1872, 618], [1922, 618], [1922, 581]], np.int32),
    "241": np.array([[1872, 618], [1872, 655], [1922, 655], [1922, 618]], np.int32),
    "242": np.array([[1872, 655], [1872, 692], [1922, 692], [1922, 655]], np.int32),
    "243": np.array([[1872, 692], [1872, 729], [1922, 729], [1922, 692]], np.int32),
    "244": np.array([[1872, 729], [1872, 766], [1922, 766], [1922, 729]], np.int32),
    "245": np.array([[1872, 766], [1872, 803], [1922, 803], [1922, 766]], np.int32),
    "246": np.array([[1872, 803], [1872, 840], [1922, 840], [1922, 803]], np.int32),
    "247": np.array([[1872, 840], [1872, 877], [1922, 877], [1922, 840]], np.int32),
    "248": np.array([[1872, 877], [1872, 914], [1922, 914], [1922, 877]], np.int32),
    "249": np.array([[1872, 914], [1872, 951], [1922, 951], [1922, 914]], np.int32),
    "250": np.array([[1872, 973], [1872, 1008], [1922, 1008], [1922, 973]], np.int32),
    "251": np.array([[1872, 1008], [1872, 1043], [1922, 1043], [1922, 1008]], np.int32),
    "252": np.array([[1872, 1043], [1872, 1078], [1922, 1078], [1922, 1043]], np.int32),
    "253": np.array([[50, 300], [50, 345], [134, 345], [134, 300]], np.int32),
    "254": np.array([[50, 345], [50, 382], [134, 382], [134, 345]], np.int32),
    "255": np.array([[50, 382], [50, 419], [134, 419], [134, 382]], np.int32),
    "256": np.array([[50, 419], [50, 456], [134, 456], [134, 419]], np.int32),
    "257": np.array([[50, 456], [50, 491], [134, 491], [134, 456]], np.int32),
    "258": np.array([[50, 491], [50, 526], [134, 526], [134, 491]], np.int32),
    "259": np.array([[50, 526], [50, 561], [134, 561], [134, 526]], np.int32),
    "260": np.array([[50, 561], [50, 598], [134, 598], [134, 561]], np.int32),
    "261": np.array([[50, 598], [50, 635], [134, 635], [134, 598]], np.int32),
    "262": np.array([[50, 635], [50, 672], [134, 672], [134, 635]], np.int32),
    "263": np.array([[50, 672], [50, 709], [134, 709], [134, 672]], np.int32),
    "264": np.array([[50, 709], [50, 746], [134, 746], [134, 709]], np.int32),
    "265": np.array([[50, 746], [50, 783], [134, 783], [134, 746]], np.int32),
    "266": np.array([[47, 848], [72, 875], [136, 815], [111, 787]], np.int32),
    "267": np.array([[72, 875], [97, 902], [161, 842], [136, 815]], np.int32),
    "268": np.array([[97, 902], [122, 929], [187, 869], [161, 842]], np.int32),
    "269": np.array([[122, 929], [147, 956], [212, 896], [187, 869]], np.int32),
    "270": np.array([[147, 956], [172, 983], [236, 923], [211, 895]], np.int32),
    "271": np.array([[172, 983], [197, 1010], [261, 950], [236, 923]], np.int32),
    "272": np.array([[197, 1010], [222, 1037], [287, 977], [261, 950]], np.int32),
    "273": np.array([[222, 1037], [247, 1064], [312, 1004], [287, 977]], np.int32),
    "274": np.array([[247, 1064], [273, 1091], [337, 1031], [312, 1004]], np.int32),
    "275": np.array([[273, 1091], [298, 1118], [362, 1058], [337, 1031]], np.int32),
}

ZONA_INTERES = np.array([
    [899, 150], [1596, 157], [1596, 242], [1747, 251], [1746, 422], [1898, 429], [1896, 531], [1919, 533], [1919, 1074], [258, 1078], [37, 825], [37, 760], [44, 418], [53, 297], [129, 290], [133, 215], [899, 222], 
], np.int32)

# Estructura para la API
datos_api = {
    "lugares": [], 
    "vehiculos": [],
    "correo_solicitado": ""
}

# ==========================================
# 2. MOTOR DE VISIÓN INVISIBLE (NUEVO)
# ==========================================
def procesar_video():
    global datos_api
    modelo = YOLO(MODEL_PATH)
    
    cap = cv2.VideoCapture(VIDEO_PATH)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    tiempo_por_frame = (1.0 / fps) / VELOCIDAD_REPRODUCCION if fps > 0 else 0.033
    video_out = cv2.VideoWriter(OUTPUT_PATH, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    while cap.isOpened():
        inicio_loop = time.time()
        
        ret, frame = cap.read()
        if not ret: 
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        resultados = modelo(frame, verbose=False, conf=UMBRAL_CONFIANZA, classes=[1, 2, 3, 4, 5, 6, 7], iou=0.4)[0]
        
        # 1. Empaquetar las coordenadas de los lugares
        lugares_json = []
        for nombre_lugar, puntos in LUGARES.items():
            lugares_json.append(puntos.tolist())
        
        vehiculos_detectados = []

        # 2. EVALUAR CARROS 
        for box in resultados.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            # FILTRO DE ZONA
            if cv2.pointPolygonTest(ZONA_INTERES, (cx, cy), measureDist=False) < 0:
                continue 

            # Caja cruda a la lista para LabVIEW
            vehiculos_detectados.append([x1, y1, x2, y2])

        # 3. GUARDAR EL FRAME TOTALMENTE LIMPIO
        ruta_temporal = IMAGEN_SALIDA.replace(".jpg", "_temp.jpg")
        
        cv2.imwrite(ruta_temporal, frame)
        
        try:
            # Se intenta reemplazar la imagen. Si LabVIEW no la está usando, será instantáneo.
            os.replace(ruta_temporal, IMAGEN_SALIDA)
        except PermissionError:
            # Si LabVIEW la está leyendo en este microsegundo, se ignroa y se pasa. 
            pass 
        
        video_out.write(frame)

        # 4. ACTUALIZAR API
        datos_api["lugares"] = lugares_json
        datos_api["vehiculos"] = vehiculos_detectados

        # Control de velocidad
        if VELOCIDAD_REPRODUCCION > 0:
            tiempo_procesamiento = time.time() - inicio_loop
            tiempo_espera = tiempo_por_frame - tiempo_procesamiento
            if tiempo_espera > 0:
                time.sleep(tiempo_espera)

    cap.release()
    video_out.release()

# ==========================================
# 3. SERVIDOR API WEB (Flask)
# ==========================================
app = Flask(__name__)

# RUTA 1: LA QUE LEE LABVIEW 
@app.route('/api', methods=['GET'])
def obtener_datos():
    # entregamos los datos de forma segura
    return jsonify(datos_api)
    
# ==========================================
# 4. PARTE DEL QR
# ==========================================
base_de_datos_boletos = {}
IP_COMPUTADORA = "192.168.100.25" 

@app.route('/generar_boleto', methods=['GET'])
def generar_boleto():
    id_boleto = "QR-" + str(uuid.uuid4())[:4].upper()
    
    # Obtenemos la fecha y hora exacta actual en formato estándar (Año-Mes-Día Hora:Minuto:Segundo)
    ahora = datetime.now()
    hora_entrada_str = ahora.strftime("%Y-%m-%d %H:%M:%S")
    
    base_de_datos_boletos[id_boleto] = {
        "hora_entrada": hora_entrada_str,
        "pagado": False
    }
    
    link_web = f"https://script-guileless-thousand.ngrok-free.dev/boleto/{id_boleto}"
    
    img_qr = qrcode.make(link_web).convert('RGB')
    ruta_absoluta_qr = "C:/Users/choco/Documents/Python/Labview_Estacionamiento/qr_actual.png"
    img_qr.save(ruta_absoluta_qr)
    
    # AGREGAMOS LA HORA AL JSON QUE RECIBE LABVIEW
    return jsonify({
        "mensaje": "QR Generado", 
        "id": id_boleto,
        "hora_entrada": hora_entrada_str
    })


@app.route('/boleto/<id_boleto>', methods=['GET'])
def ver_boleto(id_boleto):
    if id_boleto in base_de_datos_boletos:
        datos = base_de_datos_boletos[id_boleto]
        
        url_imagen_qr = f"https://api.qrserver.com/v1/create-qr-code/?size=200x200&data={id_boleto}"
        
        html = f"""
        <html>
            <head>
                <meta name="viewport" content="width=device-width, initial-scale=1.0"> 
            </head>
            <body style="text-align: center; font-family: Arial; background-color: #f4f4f9; padding: 20px;">
                <div style="background: white; padding: 30px; border-radius: 15px; display: inline-block; box-shadow: 0px 8px 15px rgba(0,0,0,0.1); max-width: 350px;">
                    <h1 style="color: #2c3e50; margin-bottom: 5px;">Ticket Virtual</h1>
                    <p style="color: #7f8c8d; font-size: 14px; margin-top: 0;">Conserve este código para pagar</p>
            
                    <h2 style="font-size: 32px; color: #34495e; margin: 20px 0;">ID: {id_boleto}</h2>
            
                    <div style="background-color: #e8f8f5; border-left: 5px solid #1abc9c; padding: 10px; margin-bottom: 20px;">
                        <h3 style="color: #16a085; margin: 0;">Hora de Entrada:</h3>
                        <p style="font-size: 24px; font-weight: bold; margin: 5px 0;">{datos['hora_entrada']}</p>
                    </div>
            
                    <img src="{url_imagen_qr}" alt="QR Salida" style="border: 2px solid #bdc3c7; border-radius: 10px; padding: 10px; margin-top: 10px; margin-bottom: 20px; width: 200px;">

                    <hr style="border: 0; border-top: 1px solid #ecf0f1; margin: 15px 0;">
                    <p style="color: #34495e; font-size: 14px; font-weight: bold; margin-bottom: 10px;">¿Necesitas ver los lugares libres?</p>
            
                    <div style="margin: 0;">
                        <input type="email" id="emailInput" required placeholder="tu@correo.com" 
                            style="padding: 10px; border: 1px solid #bdc3c7; border-radius: 5px; width: 90%; margin-bottom: 10px; box-sizing: border-box; font-family: Arial;">
                        <br>
                        <button onclick="mandarCorreo()" 
                                style="background-color: #2c3e50; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; font-weight: bold; font-family: Arial; width: 90%;">
                            Enviar Mapa a mi Correo
                        </button>
                        
                        <p id="mensajeExito" style="color: #16a085; font-weight: bold; font-size: 14px; margin-top: 10px; height: 15px;"></p>
                    </div>

                    <script>
                    function mandarCorreo() {{
                        let correo = document.getElementById("emailInput").value;
                        
                        // Si le pica sin escribir nada, no hacemos nada
                        if(correo === "") return; 
                        
                        // Mandamos el correo por debajo del agua a tu ruta /enviar_foto
                        fetch('/enviar_foto', {{
                            method: 'POST',
                            headers: {{ 'Content-Type': 'application/x-www-form-urlencoded' }},
                            body: new URLSearchParams({{'email': correo}})
                        }});

                        // Borramos la caja
                        document.getElementById("emailInput").value = '';
                        
                        // Mostramos el texto de exito en verde
                        document.getElementById("mensajeExito").innerText = '¡Correo enviado exitosamente!';
                    }}
                    </script>
                    
                </div>
            </body>
        </html>
        """
        return html
    else:
        return "Boleto no encontrado o ya pagado", 404
    

@app.route('/enviar_foto', methods=['POST'])
def recibir_correo_del_celular():
    global datos_api
    
    # 1. Sacamos el texto que escribio el usuario
    correo_recibido = request.form.get('email', '')
    
    # 2. Lo inyectamos en la memoria para que LabVIEW lo vea
    datos_api["correo_solicitado"] = correo_recibido
    print(f"NUEVO CORREO RECIBIDO: {correo_recibido}")
    
    # 3. EL BORRADO MÁGICO CON RETRASO
    # funcion que limpia la memoria
    def limpiar_memoria():
        datos_api["correo_solicitado"] = ""
        print("Memoria borrada, listo para el siguiente.")
        
    # 
    threading.Timer(3.0, limpiar_memoria).start()
    
    # 
    return "OK", 200


if __name__ == '__main__':
    hilo_vision = threading.Thread(target=procesar_video)
    hilo_vision.daemon = True
    hilo_vision.start()
    print("API CORRIENDO EN: http://127.0.0.1:5050/api")
    app.run(host='0.0.0.0', port=5050, debug=False, use_reloader=False)