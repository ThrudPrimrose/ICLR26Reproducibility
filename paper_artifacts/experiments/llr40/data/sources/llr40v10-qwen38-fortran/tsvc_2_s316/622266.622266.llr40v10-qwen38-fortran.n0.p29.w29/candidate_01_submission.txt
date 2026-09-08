subroutine tsvc_2_s316_fp64(a, result, len_1d) bind(C, name="tsvc_2_s316_fp64")
  use, intrinsic :: iso_c_binding
  use, intrinsic :: omp_lib
  implicit none
  type(c_ptr), value :: a
  type(c_ptr), value :: result
  integer(c_int64_t), value :: len_1d

  real(c_double), pointer :: av(:), rv(:)
  real(c_double) :: x
  integer(kind=8) :: n, lo, hi, ch
  integer :: nt, t

  n = len_1d
  if (n < 1) return
  call c_f_pointer(a, av, [n])
  call c_f_pointer(result, rv, [1])

  x = av(1)
  nt = 5
  if (n < 4000000) then
     x = minval(av)
  else
     ch = (n + nt - 1) / nt
!$omp parallel shared(av,n,ch) reduction(min:x) num_threads(nt)
        t = omp_get_thread_num()
        if (t == 0) then
           lo = 1
        else
           lo = t * ch + 1
        end if
        hi = n
        if (lo < n) hi = min(n, lo + ch - 1)
        if (lo <= hi) x = minval(av(lo:hi))
!$omp end parallel
  end if
  rv(1) = x
end subroutine tsvc_2_s316_fp64
