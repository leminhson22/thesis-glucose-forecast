**Bảng 3.X. Tốc độ thay đổi glucose tuyệt đối trong mỗi khoảng 5 phút**

| Cách tổng hợp | P50 | P90 | P95 | P99 | Diễn giải |
|---|---:|---:|---:|---:|---|
| Gộp toàn bộ khoảng 5 phút<br>(row-weighted) | 1,67 | 6,67 | 9,00 | 17,00 | Phản ánh phân bố của toàn bộ các khoảng 5 phút; bệnh nhân có thời gian ghi nhận dài đóng góp nhiều hơn. |
| Trung bình của phân vị từng bệnh nhân | 2,42 | 8,87 | 11,77 | 19,82 | Mỗi bệnh nhân được tính phân vị riêng trước, sau đó lấy trung bình trên 25 bệnh nhân. |
| Trung vị của phân vị từng bệnh nhân | 2,33 | 7,67 | 10,00 | 16,00 | Mô tả bệnh nhân điển hình hơn và ít bị ảnh hưởng bởi bệnh nhân có biến động quá lớn. |

*Ghi chú.* Các giá trị có đơn vị mg/dL mỗi 5 phút và được tính từ `|glucose(t) - glucose(t-1)|` sau khi sắp xếp bản ghi của từng bệnh nhân theo thời gian. P50 là trung vị, P90/P95/P99 lần lượt là các phân vị 90, 95 và 99.
