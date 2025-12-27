% Defense of Conv-Only SR-Adam Strategy

## ۱- آیا انتخاب Conv-only قابل دفاع است؟

### بله، کاملاً قابل دفاع است. سه دلیل نظری:

#### ۱. شرط ابعاد Stein-Rule (p ≥ 3)
- قضیه James-Stein تنها برای p ≥ 3 ثابت شده است
- Conv layers دارای هزاران پارامتر در فضای چندبعدی هستند
- Linear-2 فقط 10 بُعد دارد → شرط Stein ضعیف‌تر است
- Linear-1 دارای 524,416 پارامتر است، اما آنها در یک تصویر کم‌تر فشرده‌شده‌اند

#### ۲. نسبت سیگنال به نویز (SNR)
Conv layers:
- نقشه‌های Feature بزرگ (32×32, 64×16×16) → gradient‌های با نویز بیشتر
- Batch normalization ندارند → انتشار نویز بیشتر
- → Stein-rule shrinkage مفید است

FC layers:
- تغییرات کمتر از بیتچ به بیتچ
- اتصال مستقیم به loss
- → Adam معمولی کافی است

#### ۳. تطابق با تئوری
- Stein-rule برای تخمین میانگین در فضاهای بالا‌بُعد است
- Conv: درخت تصمیم محدود، دامنه بزرگ
- FC: پروجکشن کم‌بُعدی → آنجا Stein برتری کمتری دارد

---

## ۲- آیا محتوای sradam_grouping_content به مقاله اضافه شده؟

**بله!** ✓ محتوا به‌طور موفق در مقاله قرار گرفته است:
- فایل: `paper_figures/paper-draft.tex` (خط 80-82)
- جدول Table \ref{tab:sradam_grouping} اضافه شده
- توضیحات کامل در پاراگراف "Rationale for Conv-Only Stein-Rule Application"

---

### جزئیات جدول:

| Layer Group | Parameters | Stein-Rule |
|---|---|---|
| Conv2d-1 | 896 | ✓ |
| Conv2d-2 | 18,496 | ✓ |
| **Conv Total** | **18,528** | **3.4%** |
| Linear-1 | 524,416 | ✗ |
| Linear-2 | 1,290 | ✗ |
| **FC Total** | **525,706** | **96.6%** |
| **TOTAL** | **545,098** | — |
