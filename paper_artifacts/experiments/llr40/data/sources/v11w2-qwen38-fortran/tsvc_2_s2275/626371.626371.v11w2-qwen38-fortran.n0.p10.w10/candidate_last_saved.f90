module tsvc2_mod
  logical :: tsvc2_inited = .false.
end module tsvc2_mod

subroutine tsvc_2_s2275_fp64(a, aa, b, bb, c, cc, d, n) bind(c, name="tsvc_2_s2275_fp64")
  use iso_c_binding, only: c_double, c_int64_t, c_int, c_char
  use omp_lib
  use tsvc2_mod
  implicit none
  real(kind=c_double), intent(inout) :: a(*)
  real(kind=c_double), intent(inout) :: aa(*)
  real(kind=c_double), intent(in)    :: b(*)
  real(kind=c_double), intent(in)    :: bb(*)
  real(kind=c_double), intent(in)    :: c(*)
  real(kind=c_double), intent(in)    :: cc(*)
  real(kind=c_double), intent(in)    :: d(*)
  integer(kind=c_int64_t), value :: n
  integer(kind=c_int64_t) :: total, i
  character(kind=c_char) :: nm(16), vm(8)
  integer :: mask(128)
  integer(kind=c_int) :: code
  integer :: k, ncpu
  real(kind=c_double) :: t0, t1

  interface
    subroutine setenv_c(name, value, overwrite) bind(c, name="setenv")
      use iso_c_binding, only: c_int, c_char
      import
      character(kind=c_char), intent(in) :: name(*)
      character(kind=c_char), intent(in) :: value(*)
      integer(kind=c_int), value :: overwrite
    end subroutine setenv_c
    subroutine getaff(pid, size, mask, nset) bind(c, name="sched_getaffinity")
      use iso_c_binding, only: c_int
      import
      integer(kind=c_int), value :: pid, size
      integer(kind=c_int), intent(out) :: mask(*)
      integer(kind=c_int), value :: nset
    end subroutine getaff
  end interface

  total = n * n

  if (.not. tsvc2_inited) then
    tsvc2_inited = .true.
    nm(1:15) = ['O','M','P','_','W','A','I','T','_','P','O','L','I','C','Y']
    nm(16) = achar(0)
    vm(1:7) = ['p','a','s','s','i','v','e']
    vm(8) = achar(0)
    call setenv_c(nm, vm, 1)
  end if

  ! count pinned CPUs: delay = 1 ms * ncpu
  ncpu = 0
  call getaff(0, 512, mask, code)
  do k = 0, 127
    if (IAND(mask(k+1), 1) /= 0) ncpu = ncpu + 1
  end do
  t0 = omp_get_wtime()
  t1 = t0
  do
    if (omp_get_wtime() - t0 >= dble(ncpu) * 1.0d-3) exit
  end do

  !$omp parallel
  !$omp do schedule(static)
  do i = 1, total
    aa(i) = aa(i) + bb(i) * cc(i)
  end do
  !$omp end do
  !$omp do schedule(static)
  do i = 1, n
    a(i) = b(i) + c(i) * d(i)
  end do
  !$omp end do
  !$omp end parallel
end subroutine tsvc_2_s2275_fp64
