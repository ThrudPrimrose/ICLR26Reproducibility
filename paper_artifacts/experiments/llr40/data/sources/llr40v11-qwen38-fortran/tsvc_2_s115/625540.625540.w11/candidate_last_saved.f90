! TSVC tsvc_2 s115: forward solve  a(i) -= aa(j,i)*a(j)  for j < i
! 1-D flat indexing of aa (row-major): aa1((j-1)*n + i) == C aa[(j-1)*n + (i-1)].
! Bit-exact vs IEEE mul-then-sub reference: product rounded separately
! from the subtract (no FMA contraction) -- the product is held in an
! intermediate array so the compiler emits separate mul and sub.
! Single persistent OpenMP team; one worksharing do per j (row chunk of aa).
! C ABI: void tsvc_2_s115_fp64(double *restrict a, const double *restrict aa, int64_t LEN_2D)
subroutine tsvc_2_s115_fp64(a, aa, len2d) bind(C, name="tsvc_2_s115_fp64")
  use, intrinsic :: iso_c_binding
  implicit none
  type(c_ptr), value :: a
  type(c_ptr), value :: aa
  integer(c_int64_t), value :: len2d

  real(c_double), contiguous, pointer :: ad(:)
  real(c_double), contiguous, pointer :: aa1(:)
  real(c_double), allocatable :: tmp(:)
  integer(c_int64_t) :: n, i, j, base

  n = len2d
  call c_f_pointer(a, ad, [n])
  call c_f_pointer(aa, aa1, [n*n])
  if (n < 2) return
  allocate(tmp(n))

  if (n < 1024) then
     do j = 1, n
        base = (j - 1) * n
        do i = j + 1, n
           tmp(i) = aa1(base + i) * ad(j)
           ad(i) = ad(i) - tmp(i)
        end do
     end do
  else
     !$omp parallel default(none) shared(aa1, ad, tmp, n)
        do j = 1, n
           base = (j - 1) * n
           !$omp do shared(aa1, ad, tmp, n, base)
           do i = j + 1, n
              tmp(i) = aa1(base + i) * ad(j)
              ad(i) = ad(i) - tmp(i)
           end do
        end do
     !$omp end parallel
  end if
end subroutine tsvc_2_s115_fp64
