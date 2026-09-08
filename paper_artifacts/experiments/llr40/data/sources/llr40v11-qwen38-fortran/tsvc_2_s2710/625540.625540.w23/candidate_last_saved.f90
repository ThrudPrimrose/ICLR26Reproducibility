module tsvc_mod
  use iso_c_binding
  use omp_lib
  implicit none
  real(c_double), allocatable :: pa(:), pb(:), pc(:), pd(:), pe(:)
  integer :: best_t = 4
  logical, save :: ready = .false.
  integer, parameter :: P = 131072
  integer, parameter :: TC(4) = [1, 2, 8, 24]
contains

  subroutine ensure_probe()
    integer :: k
    real(c_double) :: t0, t1, bt
    if (ready) return
    if (.not. allocated(pa)) then
       allocate(pa(P), pb(P), pc(P), pd(P), pe(P))
       pa = 0.0d0; pb = 0.0d0; pc = 0.0d0; pd = 0.0d0; pe = 0.0d0
    end if
    bt = huge(bt)
    do k = 1, 4
       t0 = omp_get_wtime()
       call kern_big_neg(pa, pb, pc, pd, pe, P, TC(k))
       t1 = omp_get_wtime()
       if (t1 - t0 < bt) then
          bt = t1 - t0
          best_t = TC(k)
       end if
    end do
    ready = .true.
  end subroutine ensure_probe

  subroutine kern_big_pos(a, b, c, d, e, n, nt)
    use iso_c_binding
    implicit none
    real(c_double), intent(inout) :: a(*), b(*), c(*)
    real(c_double), intent(in)    :: d(*), e(*)
    integer, intent(in) :: n, nt
    integer :: i

    !$omp parallel do schedule(static) num_threads(nt) proc_bind(close)
    do i = 1, n
       if (a(i) > b(i)) then
          a(i) = a(i) + b(i) * d(i)
          c(i) = c(i) + d(i) * d(i)
       else
          b(i) = a(i) + e(i) * e(i)
          c(i) = a(i) + d(i) * d(i)
       end if
    end do
    !$omp end parallel do
  end subroutine kern_big_pos

  subroutine kern_big_neg(a, b, c, d, e, n, nt)
    use iso_c_binding
    implicit none
    real(c_double), intent(inout) :: a(*), b(*), c(*)
    real(c_double), intent(in)    :: d(*), e(*)
    integer, intent(in) :: n, nt
    integer :: i

    !$omp parallel do schedule(static) num_threads(nt) proc_bind(close)
    do i = 1, n
       if (a(i) > b(i)) then
          a(i) = a(i) + b(i) * d(i)
          c(i) = c(i) + d(i) * d(i)
       else
          b(i) = a(i) + e(i) * e(i)
          c(i) = c(i) + e(i) * e(i)
       end if
    end do
    !$omp end parallel do
  end subroutine kern_big_neg
end module tsvc_mod

subroutine tsvc_2_s2710_fp64(a, b, c, d, e, x, len_1d) bind(c, name='tsvc_2_s2710_fp64')
  use iso_c_binding
  use tsvc_mod
  implicit none
  real(c_double), intent(inout) :: a(*)
  real(c_double), intent(inout) :: b(*)
  real(c_double), intent(inout) :: c(*)
  real(c_double), intent(in)    :: d(*)
  real(c_double), intent(in)    :: e(*)
  real(c_double), intent(in)    :: x(*)
  integer(c_int64_t), value, intent(in) :: len_1d

  integer :: n, i

  if (len_1d <= 0) return
  n = int(len_1d, 4)

  if (len_1d > 10) then
     call ensure_probe()
     if (x(1) > 0.0d0) then
        call kern_big_pos(a, b, c, d, e, n, best_t)
     else
        call kern_big_neg(a, b, c, d, e, n, best_t)
     end if
  else
     do i = 1, n
        if (a(i) > b(i)) then
           a(i) = a(i) + b(i) * d(i)
           c(i) = d(i) * e(i) + 1.0d0
        else
           b(i) = a(i) + e(i) * e(i)
           if (x(1) > 0.0d0) then
              c(i) = a(i) + d(i) * d(i)
           else
              c(i) = c(i) + e(i) * e(i)
           end if
        end if
     end do
  end if
end subroutine tsvc_2_s2710_fp64
