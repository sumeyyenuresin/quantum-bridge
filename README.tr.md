# Quantum-Bridge (Türkçe)

**Qiskit ⇄ PennyLane arasında kayıpsız devre çevirisi, yerleşik eşdeğerlik doğrulaması ve ek bağımlılık gerektirmeyen sıfır-gürültü ekstrapolasyonu (ZNE).**

Araştırmacılar zamanlarının önemli bir kısmını angarya işlere harcıyor. Bir Qiskit ansatz'ını PennyLane'in türev alabilmesi için elle yeniden yazmak, Qiskit'in little-endian bit sırasını takip etmek ve her projede hata hafifletmeyi baştan kodlamak bunların başında geliyor. Quantum-Bridge bu işlerin her birini tek satıra indiriyor ve sonucun **birebir doğru** olduğunu sayısal olarak kanıtlıyor.

```python
import qbridge as qb

node = qb.to_qnode(qiskit_devresi, observable="ZZI")   # Qiskit → türevlenebilir PennyLane QNode
qc   = qb.to_qiskit(pennylane_fonksiyonu, *parametreler)  # PennyLane → Qiskit
qb.verify(qiskit_devresi)                              # EQUIVALENT: max |hata| = 1e-16
qb.zne(bagli_devre, executor)                          # gürültüsü giderilmiş beklenen değer
```

## Öne çıkan sonuçlar

Ölçümler Qiskit 2.3.0, PennyLane 0.45.1 ve Python 3.13 ile alındı.

- **Qiskit standart kapı kütüphanesinin tamamı (56/56)** global faz dahil birebir çevriliyor. Resmî PennyLane-Qiskit eklentisinde (0.45.0) bu oran 43/56.
- **300 rastgele devrenin 300'ü** birebir eşdeğer çıkıyor. Eklentide bu sayı 12/300. Eklentideki hata, eşlemesi olmayan çok kubitli kapıların `QubitUnitary`'ye ters kubit sırasıyla gömülmesinden kaynaklanıyor. Yeniden üretim betiği: `benchmarks/plugin_wire_order_repro.py`.
- Kütüphane ansatz'larında (RealAmplitudes, EfficientSU2, ZZFeatureMap) iki araç da doğru gradyan veriyor: sonlu farka göre ≤ 1.5·10⁻¹⁰.
- **ZNE**, depolarize gürültü altında ortalama mutlak hatayı 5–9,5 kat azaltıyor. Sonuçlar PennyLane'in yerleşik ZNE'siyle 5·10⁻¹⁵'ten küçük farkla örtüşüyor.
- **QML örneği:** Qiskit'te tanımlanıp PennyLane ile eğitilen bir sınıflandırıcıda ZNE, model çıktısının ideal değerden sapmasını p₂ = 0.01'de 12 kat azaltıyor.
- **Bedeli:** önceden derleme ve yürütme, eklentiye göre 1,1–1,6 kat daha yavaş.

## Kurulum

```bash
pip install -e ".[dev]"
pytest   # 191 test
```

Ayrıntılı API, tablolar ve şekiller için İngilizce [README.md](README.md) dosyasına bakın.
