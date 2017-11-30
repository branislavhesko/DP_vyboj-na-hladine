import pandas as pd
import numpy as np




class DataLog:

	def __init__(self, filename, columns = []):
		self._file = open(filename, "w")
		self._header = False
		self._write_header(columns)



	def _write_header(self, columns = [], col_length = -1):
		"""Summary

		Args:
		    columns (list, optional): Description
		    col_length (int, optional): Description

		Returns:
		    TYPE: Description

		Raises:
		    ValueError: Description
		"""
		if columns:
			header = "\t".join(columns)

		elif col_length != -1:
			header = "\t".join([chr(65) + i for i in range(columns)])

		else:
			raise ValueError("At least on of the parameters columns || col_length should be filled!")

		self._file.write(header + "\n")
		self._header = True
		return header



	def write_line(self, line):
		if not self._header:
			self._write_header(col_length = len(line))

		self._file.write("\t".join(map(str,line)) + str("\n"))

	def write_matrix(self, matrix):
		if not self._header:
			self._write_header(col_length = matrix.shape[1])

		for i in range(matrix.shape[0]):
			self._file.write("\t".join(map(str, matrix[i,:])) + "\n")

	def __del__(self):
		self._file.close()
