! ext_war_unit : a(i) = a_orig(i+1) + b(i) for i=1..n-1 ; a(n) unchanged.
! Parallel in-place with per-thread pre-read carry + barrier, AVX-512 8-wide bulk.
subroutine ext_war_unit_fp64(a, b, len_1d) bind(C, name="ext_war_unit_fp64")
  use iso_c_binding
  use omp_lib
  implicit none
  real(c_double), intent(inout) :: a(*)
  real(c_double), intent(in) :: b(*)
  integer(c_int64_t), value, intent(in) :: len_1d
  integer(c_int64_t) :: n, i, total, L, R, s
  integer :: tid, nta, V, j
  real(c_double) :: c, av(8), bv(8)

  n = len_1d
  if (n < 2) return

  ! small n: serial (avoids parallel overhead)
  if (n < 16384) then
     do i = 1, n - 1
        a(i) = a(i+1) + b(i)
     end do
     return
  end if

  total = n - 1
  V = 8
  !$omp parallel private(tid, nta, L, R, s, i, c, av, bv, j)
     tid = omp_get_thread_num()
     nta = omp_get_num_threads()
     L = total * tid / nta + 1
     R = total * (tid + 1) / nta
     if (L <= R) then
        c = a(R + 1)
     end if
     !$omp barrier
     if (L <= R) then
        s = L
        do while (s + V - 1 <= R - 1)
           do j = 0, V - 1
              av(j+1) = a(s + 1 + j)
              bv(j+1) = b(s + j)
           end do
           do j = 0, V - 1
              a(s + j) = av(j+1) + bv(j+1)
           end do
           s = s + V
        end do
        do i = s, R - 1
           a(i) = a(i+1) + b(i)
        end do
        a(R) = c + b(R)
     end if
  !$omp end parallel
end subroutine ext_war_unit_fp64
