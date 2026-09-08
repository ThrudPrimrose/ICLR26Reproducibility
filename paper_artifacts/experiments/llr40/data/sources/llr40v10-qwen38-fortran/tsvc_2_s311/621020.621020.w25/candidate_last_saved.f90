subroutine tsvc_2_s311_fp64(a, sum_out, len_1d) bind(C, name="tsvc_2_s311_fp64")
  use, intrinsic :: iso_c_binding
  use omp_lib
  implicit none
  type(c_ptr), intent(in), value :: a
  type(c_ptr), intent(in), value :: sum_out
  integer(c_int64_t), intent(in), value :: len_1d

  real(c_double), pointer :: av(:)
  real(c_double), pointer :: sv(:)
  real(c_double) :: s, s_local
  integer(c_int64_t) :: i, t, beg, cnt, n
  integer :: nt

  n = len_1d
  call c_f_pointer(a, av, [n])
  call c_f_pointer(sum_out, sv, [1])
  s = 0.0d0
  if (n > 262144) then
     nt = omp_get_max_threads()
     !$omp parallel reduction(+:s) shared(n)
     !$omp do schedule(static) private(i, t, beg, cnt, s_local)
     do t = 1, nt
        beg = (t - 1) * n / nt + 1
        cnt = n / nt
        if (t == nt) cnt = n - (nt - 1) * n / nt
        s_local = 0.0d0
        !$omp simd reduction(+:s_local)
        do i = beg, beg + cnt - 1
           s_local = s_local + av(i)
        end do
        !$omp end simd
        s = s + s_local
     end do
     !$omp end do
     !$omp end parallel
  else
     s_local = 0.0d0
     do i = 1, n
        s_local = s_local + av(i)
     end do
     s = s + s_local
  end if
  sv(1) = s
end subroutine
