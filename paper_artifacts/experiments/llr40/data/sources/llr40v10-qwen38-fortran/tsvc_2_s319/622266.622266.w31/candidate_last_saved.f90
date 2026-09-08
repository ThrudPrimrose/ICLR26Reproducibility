subroutine tsvc_2_s319_fp64(a, b, c, d, e, len_1d) bind(C, name="tsvc_2_s319_fp64")
  use iso_c_binding
  use omp_lib
  implicit none
  real(c_double), dimension(:), pointer, intent(inout) :: a, b
  real(c_double), dimension(:), pointer, intent(in)    :: c, d, e
  integer(c_int64_t), intent(in) :: len_1d

  integer(c_int64_t) :: n, k, i0, i1
  integer :: nt, tid
  real(c_double) :: total, t
  real(c_double) :: tp(64)

  n = int(len_1d, c_int64_t)
  if (n <= 0) return

  nt = omp_get_max_threads()
  if (nt > 64) nt = 64
  if (n < 32768) nt = 1

  if (nt == 1) then
    call worker(a, b, c, d, e, int(1, c_int64_t), n, t)
    b(1) = t
    return
  end if

!$omp parallel default(shared) private(tid, i0, i1, t)
    tid = omp_get_thread_num()
    i0 = int(tid, c_int64_t) * (n / nt) + int(1, c_int64_t)
    i1 = int(tid + 1, c_int64_t) * (n / nt)
    if (tid == nt - 1) i1 = n
    call worker(a, b, c, d, e, i0, i1, t)
    tp(tid) = t
!$omp end parallel

  total = 0.0d0
  do k = 1, nt
    total = total + tp(k)
  end do
  b(1) = total

contains

  subroutine worker(a, b, c, d, e, i0, i1, t)
    use iso_c_binding
    use, intrinsic :: intrinsic, only: _mm512_loadu_pd, _mm512_storeu_pd, &
         _mm512_add_pd, _mm512_setzero_pd, _mm512_reduce_add_pd
    implicit none
    real(c_double), dimension(:), intent(inout) :: a, b
    real(c_double), dimension(:), intent(in)    :: c, d, e
    integer(c_int64_t), intent(in) :: i0, i1
    real(c_double), intent(out) :: t

    integer(c_int64_t) :: idx, i
    real(c_double) :: accv

    accv = _mm512_setzero_pd()
    do idx = i0, i1 - 8, 8
      accv = _mm512_add_pd( &
        _mm512_add_pd(accv, _mm512_add_pd( _
          _mm512_loadu_pd(c(idx - 1:idx + 6)), _mm512_loadu_pd(d(idx - 1:idx + 6)))), &
        _mm512_add_pd( &
          _mm512_loadu_pd(c(idx - 1:idx + 6)), _mm512_loadu_pd(e(idx - 1:idx + 6))))
      _mm512_storeu_pd(a(idx - 1:idx + 6), _mm512_add_pd( &
        _mm512_loadu_pd(c(idx - 1:idx + 6)), _mm512_loadu_pd(d(idx - 1:idx + 6))))
      _mm512_storeu_pd(b(idx - 1:idx + 6), _mm512_add_pd( &
        _mm512_loadu_pd(c(idx - 1:idx + 6)), _mm512_loadu_pd(e(idx - 1:idx + 6))))
    end do
    t = _mm512_reduce_add_pd(accv)
    do i = idx, i1 - 1
      a(i) = c(i) + d(i)
      b(i) = c(i) + e(i)
      t = t + a(i) + b(i)
    end do
  end subroutine worker
end subroutine tsvc_2_s319_fp64
